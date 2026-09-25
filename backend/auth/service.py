"""User registration, password verification, and signed token issuance.

Only the standard library is used. The user store (:mod:`backend.auth.db`)
is imported lazily so that ``AuthService(store=...)`` can be constructed and
tested without the database module being importable.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from backend import config

if TYPE_CHECKING:  # pragma: no cover - typing only
    from backend.auth.db import UserStore

USERNAME_RE = re.compile(r"^[a-z0-9_]{3,32}$")
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128
PBKDF2_ITERATIONS = 100_000
SALT_BYTES = 16
SECRET_BYTES = 32
TOKEN_TTL_SECONDS = 24 * 60 * 60
SECRET_FILENAME = ".liveguard_secret"

# Dummy hash material so that a login for a non-existent username takes the
# same code path (and roughly the same amount of time) as a wrong password.
_DUMMY_SALT = secrets.token_bytes(SALT_BYTES)
_DUMMY_HASH = hashlib.pbkdf2_hmac(
    "sha256", b"liveguard-dummy-password", _DUMMY_SALT, PBKDF2_ITERATIONS
)


def _b64encode(data: bytes) -> str:
    """base64url-encode without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(text: str) -> bytes:
    """Decode unpadded base64url text."""
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _load_secret() -> bytes:
    """Return the persistent token-signing secret, creating it on first use.

    The secret is stored as hex in ``<DATA_DIR>/.liveguard_secret`` so that
    issued tokens remain valid across server restarts. The file is created
    with owner-only permissions (0o600) where the platform supports them.
    """
    secret_path = Path(config.DATA_DIR) / SECRET_FILENAME
    try:
        existing = secret_path.read_text(encoding="utf-8").strip()
        if existing:
            return bytes.fromhex(existing)
    except (OSError, ValueError):
        pass  # missing or unreadable/corrupt -> (re)create below

    secret_path.parent.mkdir(parents=True, exist_ok=True)
    secret = secrets.token_bytes(SECRET_BYTES)
    try:
        fd = os.open(secret_path, os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o600)
    except FileExistsError:
        # Another instance created it first; use its secret if it is usable.
        try:
            existing = secret_path.read_text(encoding="utf-8").strip()
            if existing:
                return bytes.fromhex(existing)
        except (OSError, ValueError):
            pass
        # File exists but is empty/corrupt: rewrite it with a fresh secret so
        # the key is persisted. Returning a non-persisted secret here would
        # silently change signing keys on every restart (all sessions die).
        try:
            fd2 = os.open(secret_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
            try:
                os.write(fd2, secret.hex().encode("ascii"))
            finally:
                os.close(fd2)
        except OSError as exc:
            # Not persisted: this session's tokens will not survive restart.
            print(f"[WARN] could not persist token secret: {exc}", flush=True)
        return secret
    try:
        os.write(fd, secret.hex().encode("ascii"))
    finally:
        os.close(fd)
    return secret


class AuthService:
    """Registration, password authentication, and signed token verification."""

    def __init__(self, store: UserStore | None = None):
        if store is None:
            from backend.auth.db import UserStore  # lazy: db.py may be built elsewhere

            store = UserStore()
        self.store = store
        self._secret_key: bytes | None = None
        self._secret_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_credentials(username, password) -> tuple[bool, str, str]:
        """Return ``(ok, normalized_username, error_message)``."""
        if not isinstance(username, str) or not isinstance(password, str):
            return (False, "", "username and password are required")

        username = username.strip().lower()
        if not USERNAME_RE.match(username):
            return (
                False,
                "",
                "username must be 3-32 characters: lowercase letters, digits, underscore",
            )
        if not MIN_PASSWORD_LENGTH <= len(password) <= MAX_PASSWORD_LENGTH:
            return (
                False,
                "",
                f"password must be {MIN_PASSWORD_LENGTH}-{MAX_PASSWORD_LENGTH} characters",
            )
        return (True, username, "")

    def _create_account(self, username: str, password: str) -> tuple[bool, str]:
        """Hash + insert. Caller must have validated and checked duplicates."""
        salt_bytes = secrets.token_bytes(SALT_BYTES)
        salt = salt_bytes.hex()
        password_hash = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt_bytes, PBKDF2_ITERATIONS
        ).hex()

        if not self.store.create_user(username, password_hash, salt, role="viewer"):
            # Lost a race with a concurrent register (or the store filled up).
            if self.store.get_user(username) is not None:
                return (False, "username already exists")
            return (False, "registration is closed")
        return (True, "account created")

    def register(self, username, password) -> tuple[bool, str]:
        """Create a new account via the dashboard. Returns ``(ok, message)``.

        Subject to :meth:`registration_allowed` (bootstrap-first-user policy).
        """
        ok, username, err = self._validate_credentials(username, password)
        if not ok:
            return (False, err)

        # Cheap checks BEFORE the expensive PBKDF2 hash: a registration
        # endpoint must be able to refuse (duplicate / closed policy) without
        # burning 100k hashing iterations first -- otherwise an unauthenticated
        # client can DoS the event loop (see telemetry_server rate limits).
        if self.store.get_user(username) is not None:
            return (False, "username already exists")
        if not self.registration_allowed():
            return (
                False,
                "registration is closed - create the account with "
                "'python -m backend.auth.cli create-user <username>' "
                "or set LIVEGUARD_ALLOW_REGISTER=1",
            )
        return self._create_account(username, password)

    def provision(self, username, password) -> tuple[bool, str]:
        """Admin provisioning path (CLI): same validation, NO policy check.

        The bootstrap-closed registration policy must never block the admin
        remedy it points to, so the CLI uses this instead of :meth:`register`.
        """
        ok, username, err = self._validate_credentials(username, password)
        if not ok:
            return (False, err)
        if self.store.get_user(username) is not None:
            return (False, "username already exists")
        return self._create_account(username, password)

    def registration_allowed(self) -> bool:
        """Registration policy: bootstrap-first-user, overridable by env.

        - ``LIVEGUARD_ALLOW_REGISTER`` set to ``1/true/yes/on``  -> always open
        - set to ``0/false/no/off``                              -> always closed
        - unset                                                   -> open only
          while the store has no users (first account bootstraps the system;
          every later account is provisioned via the CLI).
        """
        flag = os.environ.get("LIVEGUARD_ALLOW_REGISTER", "").strip().lower()
        if flag in ("1", "true", "yes", "on"):
            return True
        if flag in ("0", "false", "no", "off"):
            return False
        return self.store.user_count() == 0

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------
    def authenticate(self, username, password) -> str | None:
        """Verify credentials and return a fresh token, or ``None`` on failure."""
        if not isinstance(username, str) or not isinstance(password, str):
            return None

        username = username.strip().lower()
        user = self.store.get_user(username)
        if user is None:
            # Same failure path as a wrong password (timing does not leak
            # whether the account exists).
            dummy = hashlib.pbkdf2_hmac(
                "sha256", password.encode("utf-8"), _DUMMY_SALT, PBKDF2_ITERATIONS
            )
            hmac.compare_digest(dummy, _DUMMY_HASH)
            return None

        try:
            salt_bytes = bytes.fromhex(user["salt"])
            stored_hash = user["password_hash"]
        except (KeyError, TypeError, ValueError):
            return None

        candidate = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt_bytes, PBKDF2_ITERATIONS
        ).hex()
        if not isinstance(stored_hash, str) or not hmac.compare_digest(
            candidate, stored_hash
        ):
            return None

        self.store.record_login(username)
        return self._issue_token(username)

    def verify_token(self, token) -> str | None:
        """Return the username carried by a valid, unexpired token, else ``None``."""
        if not isinstance(token, str) or not token.isascii():
            return None
        parts = token.split(".")
        if len(parts) != 2:
            return None
        payload_b64, signature_b64 = parts
        if not payload_b64 or not signature_b64:
            return None

        expected = _b64encode(
            hmac.new(
                self._get_secret(), payload_b64.encode("ascii"), hashlib.sha256
            ).digest()
        )
        if not hmac.compare_digest(signature_b64, expected):
            return None

        try:
            payload = json.loads(_b64decode(payload_b64))
            username = payload["u"]
            expiry = payload["e"]
        except (ValueError, KeyError, TypeError):
            return None
        if not isinstance(username, str) or not isinstance(expiry, (int, float)):
            return None
        if time.time() >= expiry:
            return None
        return username

    # ------------------------------------------------------------------
    # Token helpers
    # ------------------------------------------------------------------
    def _issue_token(self, username: str) -> str:
        return self._make_token(username, time.time() + TOKEN_TTL_SECONDS)

    def _make_token(self, username: str, expires_at: float) -> str:
        payload_json = json.dumps({"u": username, "e": int(expires_at)})
        payload_b64 = _b64encode(payload_json.encode("utf-8"))
        signature = hmac.new(
            self._get_secret(), payload_b64.encode("ascii"), hashlib.sha256
        ).digest()
        return f"{payload_b64}.{_b64encode(signature)}"

    def _get_secret(self) -> bytes:
        # Lazy, but locked: with register/authenticate now running on
        # asyncio.to_thread workers while verify_token runs on the event-loop
        # thread, two threads could race the first load and end up holding
        # different secrets (one persisted, one not) -> tokens minted in that
        # window would never verify.
        if self._secret_key is None:
            with self._secret_lock:
                if self._secret_key is None:
                    self._secret_key = _load_secret()
        return self._secret_key
