"""End-to-end validation of the LiveGuard-EHMS authentication flow (stdlib unittest).

Sections:
  A. Unit level      - UserStore round-trip (test 1), AuthService (test 2)
  B. Wire protocol   - real TelemetryServer on a free port with a real SQLite
                       store in a temp dir (tests 3-10)

LIVEGUARD_DATA_DIR is pointed at a throwaway temp directory BEFORE
backend.config is imported, so the repo's data/ directory (existing
smoke-test db + secret) is never read or written by these tests.

Run from the repo root:
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP_DATA_DIR = tempfile.mkdtemp(prefix="liveguard_auth_tests_")
os.environ["LIVEGUARD_DATA_DIR"] = TMP_DATA_DIR  # BEFORE backend.config import
# Registration policy: default is bootstrap-first-user only, which would break
# tests that exercise registration *mechanics* (they register many users per
# store). Keep registration open module-wide; RegistrationPolicyTests toggles
# this env var to verify the actual policy.
os.environ["LIVEGUARD_ALLOW_REGISTER"] = "1"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend import config  # noqa: E402
from backend import telemetry_server as ts_module  # noqa: E402
from backend.auth.db import UserStore  # noqa: E402
from backend.auth.service import AuthService  # noqa: E402
from backend.telemetry_server import TelemetryServer  # noqa: E402
from websockets.exceptions import ConnectionClosed  # noqa: E402
from websockets.sync.client import connect  # noqa: E402

GOOD_PASSWORD = "s3cret-pass"  # 11 chars -> satisfies the 8-128 rule


def tearDownModule():
    shutil.rmtree(TMP_DATA_DIR, ignore_errors=True)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def free_port() -> int:
    """Bind an OS-assigned free port, release it, return the number."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def send_json(ws, payload: dict) -> None:
    ws.send(json.dumps(payload))


def recv_json(ws, timeout: float = 2.0):
    """Next frame as a dict, or None if nothing arrived before the timeout."""
    try:
        raw = ws.recv(timeout=timeout)
    except TimeoutError:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    return json.loads(raw)


def recv_close(ws, timeout: float = 5.0):
    """Read until the connection closes; return (close_code, close_reason)."""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError("connection was not closed within %.1fs" % timeout)
        try:
            ws.recv(timeout=remaining)
        except ConnectionClosed as exc:
            close = exc.rcvd if exc.rcvd is not None else exc.sent
            if close is None:
                return (None, "")
            return (close.code, close.reason)
        except TimeoutError:
            raise AssertionError("connection was not closed within %.1fs" % timeout)


def wait_until_ready(url: str, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            with connect(url, open_timeout=1):
                return
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def read_last_login(db_path: str, username: str):
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT last_login_at FROM users WHERE username = ?", (username,)
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


# ======================================================================
# A1. UserStore
# ======================================================================
class RegistrationPolicyTests(unittest.TestCase):
    """Registration policy: open only for the first (bootstrap) user.

    ``LIVEGUARD_ALLOW_REGISTER`` overrides it (1/0); unset = bootstrap-only.
    """

    def setUp(self):
        self._saved = os.environ.pop("LIVEGUARD_ALLOW_REGISTER", None)
        self.dir = tempfile.mkdtemp(prefix="lg_policy_", dir=TMP_DATA_DIR)
        self.store = UserStore(db_path=os.path.join(self.dir, "policy.db"))
        self.auth = AuthService(store=self.store)

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.dir, ignore_errors=True)
        if self._saved is None:
            os.environ.pop("LIVEGUARD_ALLOW_REGISTER", None)
        else:
            os.environ["LIVEGUARD_ALLOW_REGISTER"] = self._saved

    def test_first_user_bootstraps_then_closed(self):
        ok, msg = self.auth.register("first_user", GOOD_PASSWORD)
        self.assertTrue(ok, msg)
        # duplicate check still wins over the closed-policy message
        ok, msg = self.auth.register("first_user", GOOD_PASSWORD)
        self.assertFalse(ok)
        self.assertEqual(msg, "username already exists")
        # everything after the bootstrap user is closed by default
        ok, msg = self.auth.register("second_user", GOOD_PASSWORD)
        self.assertFalse(ok)
        self.assertIn("registration is closed", msg)
        self.assertEqual(self.store.user_count(), 1)

    def test_env_override_open(self):
        os.environ["LIVEGUARD_ALLOW_REGISTER"] = "1"
        self.assertTrue(self.auth.register("first_user", GOOD_PASSWORD)[0])
        self.assertTrue(self.auth.register("second_user", GOOD_PASSWORD)[0])
        self.assertEqual(self.store.user_count(), 2)

    def test_env_override_closed_blocks_bootstrap(self):
        os.environ["LIVEGUARD_ALLOW_REGISTER"] = "0"
        ok, msg = self.auth.register("first_user", GOOD_PASSWORD)
        self.assertFalse(ok)
        self.assertIn("registration is closed", msg)
        self.assertEqual(self.store.user_count(), 0)
        # ...but the admin remedy (CLI -> provision) always works:
        ok, msg = self.auth.provision("admin_user", GOOD_PASSWORD)
        self.assertTrue(ok, msg)
        self.assertEqual(self.store.user_count(), 1)
        # provision still enforces validation and duplicate checks
        ok, msg = self.auth.provision("admin_user", GOOD_PASSWORD)
        self.assertFalse(ok)
        self.assertEqual(msg, "username already exists")
        ok, msg = self.auth.provision("bad name", GOOD_PASSWORD)
        self.assertFalse(ok)
        self.assertEqual(self.store.user_count(), 1)


class CliTests(unittest.TestCase):
    """`python -m backend.auth.cli create-user` (provision path)."""

    def _run_cli(self, username: str, stdin_text: str):
        from backend.auth import cli

        old_stdin = sys.stdin
        sys.stdin = io.StringIO(stdin_text)  # not a tty -> line-input fallback
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                rc = cli.main(["create-user", username])
        finally:
            sys.stdin = old_stdin
        return rc, buf.getvalue()

    def _run_cli_args(self, argv, stdin_text: str = ""):
        """Run the CLI with arbitrary argv; returns (exit_code, captured_output)."""
        from backend.auth import cli

        old_stdin = sys.stdin
        sys.stdin = io.StringIO(stdin_text)  # not a tty -> line-input fallback
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                rc = cli.main(argv)
        finally:
            sys.stdin = old_stdin
        return rc, buf.getvalue()

    @contextlib.contextmanager
    def _isolated_data_dir(self):
        """Point config.DATA_DIR at a throwaway dir for this test.

        The CLI always opens the *default* store (AuthService() with no
        argument), which resolves config.DATA_DIR at construction time --
        so patching it gives each test a private, cleanup-able user store
        without touching the shared default one.
        """
        saved = config.DATA_DIR
        isolated = tempfile.mkdtemp(prefix="lg_cli_", dir=TMP_DATA_DIR)
        config.DATA_DIR = isolated
        try:
            yield isolated
        finally:
            config.DATA_DIR = saved
            shutil.rmtree(isolated, ignore_errors=True)

    def test_cli_create_user_from_piped_stdin(self):
        rc, out = self._run_cli("cli_user", "cli-pass-123\ncli-pass-123\n")
        self.assertEqual(rc, 0, out)
        self.assertIn("Success", out)
        self.assertIsNotNone(
            AuthService().authenticate("cli_user", "cli-pass-123"),
            "CLI-created user must be able to log in",
        )

    def test_cli_rejects_mismatched_passwords(self):
        rc, out = self._run_cli("cli_user2", "cli-pass-123\ncli-pass-456\n")
        self.assertEqual(rc, 1, out)
        self.assertIn("do not match", out)
        self.assertIsNone(AuthService().authenticate("cli_user2", "cli-pass-456"))

    def test_cli_eof_on_stdin_fails_gracefully(self):
        rc, out = self._run_cli("cli_user3", "")
        self.assertEqual(rc, 1, out)
        self.assertIn("no password provided", out)

    # ------------------------------------------------------------------
    # list-users
    # ------------------------------------------------------------------
    def test_cli_list_users_empty_store(self):
        with self._isolated_data_dir():
            rc, out = self._run_cli_args(["list-users"])
        self.assertEqual(rc, 0, out)
        self.assertIn("No users found.", out)

    def test_cli_list_users_shows_created_account(self):
        with self._isolated_data_dir():
            rc, out = self._run_cli_args(
                ["create-user", "list_user"], "list-pass-123\nlist-pass-123\n"
            )
            self.assertEqual(rc, 0, out)
            self.assertIsNotNone(
                AuthService().authenticate("list_user", "list-pass-123"),
                "setup login failed",
            )  # stamps last_login_at
            rc, out = self._run_cli_args(["list-users"])
            user = AuthService().store.get_user("list_user")
        self.assertEqual(rc, 0, out)
        self.assertIn("list_user", out)
        self.assertIn("viewer", out)  # role column
        self.assertNotIn("never", out)  # last login was stamped, not missing
        # password material must NEVER be printed
        self.assertNotIn(user["password_hash"], out)
        self.assertNotIn(user["salt"], out)
        self.assertNotIn("password", out.lower())

    # ------------------------------------------------------------------
    # change-password
    # ------------------------------------------------------------------
    def test_cli_change_password_success(self):
        with self._isolated_data_dir():
            rc, out = self._run_cli_args(
                ["create-user", "chg_user"], "old-pass-123\nold-pass-123\n"
            )
            self.assertEqual(rc, 0, out)
            rc, out = self._run_cli_args(
                ["change-password", "chg_user"],
                "old-pass-123\nnew-pass-456\nnew-pass-456\n",
            )
            self.assertEqual(rc, 0, out)
            self.assertIn("Success", out)
            self.assertIsNone(
                AuthService().authenticate("chg_user", "old-pass-123"),
                "old password must no longer work",
            )
            self.assertIsNotNone(
                AuthService().authenticate("chg_user", "new-pass-456"),
                "login with the new password must work",
            )

    def test_cli_change_password_wrong_current(self):
        with self._isolated_data_dir():
            rc, out = self._run_cli_args(
                ["create-user", "chg_wrong"], "old-pass-123\nold-pass-123\n"
            )
            self.assertEqual(rc, 0, out)
            rc, out = self._run_cli_args(
                ["change-password", "chg_wrong"],
                "not-the-current\nnew-pass-456\nnew-pass-456\n",
            )
            self.assertEqual(rc, 1, out)
            self.assertIn("current password is incorrect", out)
            # the stored password is untouched
            self.assertIsNotNone(AuthService().authenticate("chg_wrong", "old-pass-123"))
            self.assertIsNone(AuthService().authenticate("chg_wrong", "new-pass-456"))

    def test_cli_change_password_mismatched_new(self):
        with self._isolated_data_dir():
            rc, out = self._run_cli_args(
                ["create-user", "chg_mismatch"], "old-pass-123\nold-pass-123\n"
            )
            self.assertEqual(rc, 0, out)
            rc, out = self._run_cli_args(
                ["change-password", "chg_mismatch"],
                "old-pass-123\nnew-pass-456\nnew-pass-789\n",
            )
            self.assertEqual(rc, 1, out)
            self.assertIn("do not match", out)
            self.assertIsNotNone(
                AuthService().authenticate("chg_mismatch", "old-pass-123")
            )
            self.assertIsNone(AuthService().authenticate("chg_mismatch", "new-pass-456"))

    def test_cli_change_password_nonexistent_user(self):
        with self._isolated_data_dir():
            rc, out = self._run_cli_args(
                ["change-password", "ghost_user"],
                "whatever-pass\nnew-pass-456\nnew-pass-456\n",
            )
            self.assertEqual(rc, 1, out)
            self.assertIn("user not found", out)
            self.assertEqual(AuthService().store.user_count(), 0)

    # ------------------------------------------------------------------
    # delete-user
    # ------------------------------------------------------------------
    def test_cli_delete_user_removes_account(self):
        with self._isolated_data_dir():
            rc, out = self._run_cli_args(
                ["create-user", "del_user"], "del-pass-123\ndel-pass-123\n"
            )
            self.assertEqual(rc, 0, out)
            rc, out = self._run_cli_args(
                ["create-user", "keep_user"], "keep-pass-123\nkeep-pass-123\n"
            )
            self.assertEqual(rc, 0, out)
            rc, out = self._run_cli_args(["delete-user", "del_user"])
            self.assertEqual(rc, 0, out)
            self.assertIn("Success", out)
            self.assertIsNone(AuthService().store.get_user("del_user"))
            self.assertIsNone(AuthService().authenticate("del_user", "del-pass-123"))
            # the sibling account is untouched
            self.assertIsNotNone(AuthService().store.get_user("keep_user"))
            self.assertIsNotNone(AuthService().authenticate("keep_user", "keep-pass-123"))

    def test_cli_delete_last_user_refused(self):
        with self._isolated_data_dir():
            rc, out = self._run_cli_args(
                ["create-user", "solo_user"], "solo-pass-123\nsolo-pass-123\n"
            )
            self.assertEqual(rc, 0, out)
            rc, out = self._run_cli_args(["delete-user", "solo_user"])
            self.assertEqual(rc, 1, out)
            self.assertIn("last remaining user", out)
            self.assertIsNotNone(AuthService().store.get_user("solo_user"))
            self.assertIsNotNone(AuthService().authenticate("solo_user", "solo-pass-123"))

    def test_cli_delete_nonexistent_user(self):
        with self._isolated_data_dir():
            rc, out = self._run_cli_args(
                ["create-user", "real_user"], "real-pass-123\nreal-pass-123\n"
            )
            self.assertEqual(rc, 0, out)
            rc, out = self._run_cli_args(["delete-user", "ghost_user"])
            self.assertEqual(rc, 1, out)
            self.assertIn("user not found", out)
            # the real account is untouched
            self.assertIsNotNone(AuthService().store.get_user("real_user"))


class IPBudgetTests(unittest.TestCase):
    """Per-IP failed-auth budget on a dedicated server (own budget state).

    Order matters (unittest sorts alphabetically): each test leaves the
    budget in a known state for the next.
    """

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="lg_ipbudget_", dir=TMP_DATA_DIR)
        cls.store = UserStore(db_path=os.path.join(cls.dir, "budget.db"))
        cls.auth = AuthService(store=cls.store)
        ok, msg = cls.auth.provision("budget_user", GOOD_PASSWORD)
        assert ok, msg
        cls.server = TelemetryServer(host="127.0.0.1", port=free_port(), auth=cls.auth)
        cls.server.start()
        time.sleep(0.5)
        cls.url = "ws://127.0.0.1:%d" % cls.server.port

    @classmethod
    def tearDownClass(cls):
        # Close the sqlite handle first: on Windows an open budget.db blocks
        # deletion and ignore_errors=True would silently leak the directory.
        cls.store.close()
        shutil.rmtree(cls.dir, ignore_errors=True)

    def _budget(self) -> int:
        entry = self.server._ip_failures.get("127.0.0.1")
        return entry[0] if entry else 0

    def _attempt(self, password: str, expect_close: bool = False):
        """One connection, one auth attempt -> (reply_reason, close_code)."""
        with connect(self.url, open_timeout=5) as ws:
            send_json(ws, {"type": "auth", "username": "budget_user",
                           "password": password})
            msg = recv_json(ws, timeout=3)
            reason = (msg or {}).get("reason")
            code = None
            if expect_close:
                code, _close_reason = recv_close(ws, timeout=3)
            return reason, code

    def test_01_successful_login_does_not_consume_budget(self):
        before = self._budget()
        reason, _ = self._attempt(GOOD_PASSWORD)
        self.assertNotIn(reason, ("invalid credentials", "too many attempts"))
        time.sleep(0.2)  # let the refund land (it precedes auth_ok, but be safe)
        self.assertEqual(self._budget(), before)

    def test_02_budget_exhaustion_rate_limits_even_correct_password(self):
        for _ in range(10):
            reason, _ = self._attempt("wrong-password")
            self.assertEqual(reason, "invalid credentials")
        self.assertEqual(self._budget(), ts_module.MAX_IP_AUTH_FAILURES)
        # 11th attempt is pre-denied without evaluating the credential...
        reason, code = self._attempt("wrong-password", expect_close=True)
        self.assertEqual(reason, "too many attempts")
        self.assertEqual(code, 1008)
        # ...and this is a rate limit on the *IP*, not an account lockout:
        # it self-heals when the window expires (next test).

    def test_03_budget_window_self_heals(self):
        self.assertEqual(self._budget(), ts_module.MAX_IP_AUTH_FAILURES)
        original = ts_module.IP_FAILURE_WINDOW_SECONDS
        ts_module.IP_FAILURE_WINDOW_SECONDS = 0.3
        try:
            time.sleep(0.5)
            reason, _ = self._attempt(GOOD_PASSWORD)  # auth_ok again
            self.assertNotEqual(reason, "too many attempts")
            time.sleep(0.2)
            self.assertEqual(self._budget(), 0)  # expired entry swept + refunded
        finally:
            ts_module.IP_FAILURE_WINDOW_SECONDS = original

    def test_04_parallel_burst_cannot_exceed_cap(self):
        import threading as _threading

        results = []

        def worker():
            try:
                reason, _ = self._attempt("wrong-password")
                results.append(reason)
            except Exception as exc:  # pragma: no cover - diagnostics
                results.append("EXC:%s" % exc)

        threads = [_threading.Thread(target=worker) for _ in range(15)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        # Reserve-before-evaluate: exactly MAX slots, extras denied, never over.
        self.assertEqual(len(results), 15)
        self.assertEqual(self._budget(), ts_module.MAX_IP_AUTH_FAILURES)
        self.assertLessEqual(self._budget(), ts_module.MAX_IP_AUTH_FAILURES)
        denied = [r for r in results if r == "too many attempts"]
        failed = [r for r in results if r == "invalid credentials"]
        self.assertEqual(len(denied), 15 - ts_module.MAX_IP_AUTH_FAILURES)
        self.assertEqual(len(failed), ts_module.MAX_IP_AUTH_FAILURES)
        self.assertFalse([r for r in results if r.startswith("EXC:")], results)


class UserStoreTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="lg_store_", dir=TMP_DATA_DIR)
        self.db_path = os.path.join(self.dir, "users.db")
        self.store = UserStore(db_path=self.db_path)

    def tearDown(self):
        try:
            self.store.close()
        except RuntimeError:
            pass
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_01_create_get_roundtrip(self):
        self.assertTrue(self.store.create_user("alice", "hash1", "salt1"))
        user = self.store.get_user("alice")
        self.assertIsNotNone(user)
        self.assertEqual(
            set(user), {"id", "username", "password_hash", "salt", "role"}
        )
        self.assertEqual(user["username"], "alice")
        self.assertEqual(user["password_hash"], "hash1")
        self.assertEqual(user["salt"], "salt1")
        self.assertEqual(user["role"], "viewer")
        self.assertIsInstance(user["id"], int)
        self.assertIsNone(self.store.get_user("nobody"))
        # custom role survives the round-trip
        self.assertTrue(self.store.create_user("bob", "h", "s", role="admin"))
        self.assertEqual(self.store.get_user("bob")["role"], "admin")

    def test_01_duplicate_is_case_insensitive(self):
        self.assertTrue(self.store.create_user("alice", "h", "s"))
        self.assertFalse(self.store.create_user("ALICE", "h2", "s2"))
        self.assertFalse(self.store.create_user("Alice", "h3", "s3"))
        self.assertEqual(self.store.user_count(), 1)
        # lookup is case-insensitive; stored casing is the first one used
        self.assertEqual(self.store.get_user("ALICE")["username"], "alice")
        self.assertEqual(self.store.get_user("aLiCe")["username"], "alice")

    def test_01_invalid_args_rejected(self):
        self.assertFalse(self.store.create_user("", "h", "s"))
        self.assertFalse(self.store.create_user("u", "", "s"))
        self.assertFalse(self.store.create_user("u", "h", ""))
        self.assertFalse(self.store.create_user(None, "h", "s"))
        self.assertFalse(self.store.create_user("u", None, "s"))
        self.assertFalse(self.store.create_user("u", "h", "s", role=""))
        self.assertEqual(self.store.user_count(), 0)

    def test_01_user_count_and_custom_db_path(self):
        self.assertTrue(os.path.exists(self.db_path), "custom db file not created")
        self.assertEqual(self.store.user_count(), 0)
        self.store.create_user("bob", "hash-b", "salt-b")
        self.store.create_user("carol", "hash-c", "salt-c")
        self.assertEqual(self.store.user_count(), 2)
        # a second connection to the same custom path sees the same data
        self.store.close()
        other = UserStore(db_path=self.db_path)
        try:
            self.assertEqual(other.user_count(), 2)
            self.assertEqual(other.get_user("bob")["password_hash"], "hash-b")
        finally:
            other.close()

    def test_01_record_login(self):
        self.store.create_user("carol", "h", "s")
        self.assertIsNone(read_last_login(self.db_path, "carol"))
        self.store.record_login("carol")
        self.assertIsNotNone(read_last_login(self.db_path, "carol"))
        before = read_last_login(self.db_path, "carol")
        time.sleep(1.1)  # datetime('now') has 1s resolution
        self.store.record_login("carol")
        self.assertNotEqual(read_last_login(self.db_path, "carol"), before)
        self.store.record_login("ghost")  # unknown user: silent no-op, no raise

    def test_02_closed_store_raises(self):
        self.store.close()
        with self.assertRaises(RuntimeError):
            self.store.get_user("alice")
        with self.assertRaises(RuntimeError):
            self.store.create_user("x", "h", "s")
        with self.assertRaises(RuntimeError):
            self.store.user_count()
        # close() is idempotent (tearDown closes again)
        self.store.close()


# ======================================================================
# A2. AuthService
# ======================================================================
class AuthServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="lg_svc_", dir=TMP_DATA_DIR)
        cls.db_path = os.path.join(cls.dir, "service_users.db")
        cls.store = UserStore(db_path=cls.db_path)
        cls.auth = AuthService(store=cls.store)

    @classmethod
    def tearDownClass(cls):
        cls.store.close()
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_02_register_rejects_bad_usernames(self):
        before = self.store.user_count()
        bad = ["ab", "a b", "user!x", "x" * 33, "", "   ", "Ünïcode", None, 123, ["u"]]
        for name in bad:
            ok, msg = self.auth.register(name, GOOD_PASSWORD)
            self.assertFalse(ok, "username %r was accepted" % (name,))
            self.assertIsInstance(msg, str)
            self.assertTrue(msg)
        self.assertEqual(self.store.user_count(), before)

    def test_02_register_rejects_bad_passwords(self):
        before = self.store.user_count()
        for pw in ["1234567", "a" * 129, "", None, 12345678, ["pw"]]:
            ok, msg = self.auth.register("pwcheck_user", pw)
            self.assertFalse(ok, "password %r was accepted" % (pw,))
            self.assertIn("password", msg)
        self.assertEqual(self.store.user_count(), before)

    def test_02_register_success_and_duplicates(self):
        ok, msg = self.auth.register("svc_user1", GOOD_PASSWORD)
        self.assertTrue(ok)
        self.assertEqual(msg, "account created")
        ok, msg = self.auth.register("svc_user1", GOOD_PASSWORD)
        self.assertFalse(ok)
        self.assertEqual(msg, "username already exists")
        # case-insensitive duplicate
        ok, msg = self.auth.register("SVC_USER1", "another-pass")
        self.assertFalse(ok)
        self.assertEqual(msg, "username already exists")
        # surrounding whitespace is stripped -> same account
        ok, msg = self.auth.register("  svc_user1  ", "another-pass")
        self.assertFalse(ok)

    def test_02_authenticate_good_bad_missing(self):
        self.auth.register("svc_user2", GOOD_PASSWORD)
        self.assertIsNone(read_last_login(self.db_path, "svc_user2"))

        token = self.auth.authenticate("svc_user2", GOOD_PASSWORD)
        self.assertIsInstance(token, str)
        self.assertTrue(token)
        self.assertIsNotNone(read_last_login(self.db_path, "svc_user2"))

        self.assertIsNone(self.auth.authenticate("svc_user2", "wrong-password"))
        self.assertIsNone(self.auth.authenticate("svc_ghost", GOOD_PASSWORD))
        self.assertIsNone(self.auth.authenticate(None, GOOD_PASSWORD))
        self.assertIsNone(self.auth.authenticate("svc_user2", None))
        self.assertIsNone(self.auth.authenticate(42, GOOD_PASSWORD))

    def test_02_verify_token_variants(self):
        self.auth.register("svc_user3", GOOD_PASSWORD)
        token = self.auth.authenticate("svc_user3", GOOD_PASSWORD)
        self.assertIsInstance(token, str)

        # valid
        self.assertEqual(self.auth.verify_token(token), "svc_user3")

        # tampered payload (signature no longer matches)
        payload, sig = token.split(".")
        idx = 5
        flipped = "x" if payload[idx] != "x" else "y"
        tampered = payload[:idx] + flipped + payload[idx + 1:] + "." + sig
        self.assertIsNone(self.auth.verify_token(tampered))

        # tampered signature
        bad_sig = "y" if sig[-1] != "y" else "x"
        self.assertIsNone(self.auth.verify_token(payload + "." + sig[:-1] + bad_sig))

        # garbage
        for garbage in ["", "hello", "a.b", "a.b.c", "!!!.@@@", "tökén.x", None, 42, []]:
            self.assertIsNone(
                self.auth.verify_token(garbage), "garbage %r verified" % (garbage,)
            )

        # expired (signed with the correct secret but a past expiry)
        expired = self.auth._make_token("svc_user3", time.time() - 5)
        self.assertIsNone(self.auth.verify_token(expired))

        # not expired yet but very close: still valid (issued now +0.5s is in the past... use +60)
        soon = self.auth._make_token("svc_user3", time.time() + 60)
        self.assertEqual(self.auth.verify_token(soon), "svc_user3")

    def test_02_token_survives_new_auth_service_instances(self):
        self.auth.register("svc_user4", GOOD_PASSWORD)
        token = self.auth.authenticate("svc_user4", GOOD_PASSWORD)
        self.assertIsInstance(token, str)

        # signing secret persisted under LIVEGUARD_DATA_DIR
        secret_file = os.path.join(str(config.DATA_DIR), ".liveguard_secret")
        self.assertTrue(os.path.exists(secret_file), "secret file not persisted")

        # 1) brand-new AuthService in this process (default store -> same DATA_DIR)
        fresh = AuthService()
        self.assertEqual(fresh.verify_token(token), "svc_user4")
        fresh.store.close()  # release the default store so tmp cleanup succeeds

        # 2) brand-new OS process with the same LIVEGUARD_DATA_DIR env var
        code = (
            "import sys; sys.path.insert(0, %r); "
            "from backend.auth.service import AuthService; "
            "print(AuthService().verify_token(%r))"
        ) % (REPO_ROOT, token)
        env = dict(os.environ)
        env["LIVEGUARD_DATA_DIR"] = TMP_DATA_DIR
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=60,
            env=env, cwd=REPO_ROOT,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "svc_user4")

    def test_02_username_normalization(self):
        ok, msg = self.auth.register("  Mixed_Case9  ", GOOD_PASSWORD)
        self.assertTrue(ok, msg)
        self.assertIsNotNone(self.store.get_user("mixed_case9"))
        token = self.auth.authenticate("MIXED_CASE9", GOOD_PASSWORD)
        self.assertIsInstance(token, str, "mixed-case login failed")
        self.assertEqual(self.auth.verify_token(token), "mixed_case9")
        token2 = self.auth.authenticate(" mixed_case9 ", GOOD_PASSWORD)
        self.assertIsInstance(token2, str, "whitespace login failed")
        self.assertEqual(self.auth.verify_token(token2), "mixed_case9")


# ======================================================================
# B. Wire protocol (real TelemetryServer, free port, real SQLite store)
# ======================================================================
class WireProtocolTests(unittest.TestCase):
    """Tests 3-10. Numbered names enforce execution order (token shared 05 -> 07/08)."""

    token = None  # class-level: token issued during test_05

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp(prefix="lg_wire_", dir=TMP_DATA_DIR)
        cls.store = UserStore(db_path=os.path.join(cls.dir, "wire_users.db"))
        cls.auth = AuthService(store=cls.store)
        ok, msg = cls.auth.register("wire_user", GOOD_PASSWORD)
        assert ok, msg
        cls.port = free_port()
        cls.server = TelemetryServer(host="127.0.0.1", port=cls.port, auth=cls.auth)
        cls.server.start()
        cls.url = "ws://127.0.0.1:%d" % cls.port
        wait_until_ready(cls.url)

    @classmethod
    def tearDownClass(cls):
        cls.store.close()
        shutil.rmtree(cls.dir, ignore_errors=True)

    # -- helpers ------------------------------------------------------
    def _login(self, ws, username="wire_user", password=GOOD_PASSWORD):
        send_json(ws, {"type": "auth", "username": username, "password": password})
        msg = recv_json(ws, timeout=5)
        self.assertIsNotNone(msg, "no reply to password auth")
        self.assertEqual(msg.get("type"), "auth_ok", msg)
        self.assertEqual(msg.get("username"), username.strip().lower(), msg)
        self.assertIsInstance(msg.get("token"), str, msg)
        return msg

    def _expect_broadcast(self, ws, marker, timeout=3.0):
        payload = {
            "type": "beat", "marker": marker, "prediction": "Normal",
            "is_anomaly": False, "confidence": 98.5, "abnormal_prob": 0.015,
            "heart_rate": 72.0, "total_alerts": 0, "total_beats": 7,
            "timestamp": time.time(),
        }
        self.server.broadcast(payload)
        msg = recv_json(ws, timeout)
        self.assertIsNotNone(msg, "broadcast %r not received" % marker)
        self.assertEqual(msg, payload)
        return msg

    def _expect_no_broadcast(self, ws, wait=1.0):
        self.server.broadcast({"type": "beat", "marker": "should-not-arrive"})
        msg = recv_json(ws, wait)
        self.assertIsNone(msg, "unauthenticated client received: %r" % (msg,))

    # -- tests --------------------------------------------------------
    def test_03_unauthenticated_receives_nothing(self):
        with connect(self.url, open_timeout=5) as anon:
            with connect(self.url, open_timeout=5) as authed:
                self._login(authed)  # a real target so broadcast() actually sends
                self.server.broadcast({"type": "beat", "marker": "t03"})
                # unauthenticated connection must hear nothing within 1s
                self.assertIsNone(
                    recv_json(anon, 1.0),
                    "unauthenticated client received a broadcast",
                )
                # ...while the authenticated one does (broadcast really happened)
                msg = recv_json(authed, 3.0)
                self.assertIsNotNone(msg, "authenticated client got nothing")
                self.assertEqual(msg.get("marker"), "t03")

    def test_04_register_does_not_authenticate(self):
        with connect(self.url, open_timeout=5) as ws:
            send_json(ws, {
                "type": "register",
                "username": "wire_reg_user",
                "password": GOOD_PASSWORD,
            })
            msg = recv_json(ws, 3.0)
            self.assertIsNotNone(msg, "no reply to register")
            self.assertEqual(msg.get("type"), "register_ok", msg)
            self.assertEqual(msg.get("username"), "wire_reg_user", msg)
            self.assertNotIn("token", msg)
            # registration alone must NOT authenticate
            self._expect_no_broadcast(ws, wait=1.0)

    def test_05_password_auth_ok_then_receives_broadcast(self):
        with connect(self.url, open_timeout=5) as ws:
            # mixed case on the wire: server must normalize
            send_json(ws, {
                "type": "auth",
                "username": "WIRE_USER",
                "password": GOOD_PASSWORD,
            })
            msg = recv_json(ws, 5.0)
            self.assertIsNotNone(msg, "no reply to auth")
            self.assertEqual(msg.get("type"), "auth_ok", msg)
            self.assertEqual(msg.get("username"), "wire_user", msg)
            self.assertIsInstance(msg.get("token"), str)
            self.assertTrue(msg.get("token"))
            type(self).token = msg["token"]
            # now broadcasts flow
            self._expect_broadcast(ws, "t05")

    def test_06_three_failures_rate_limited(self):
        with connect(self.url, open_timeout=5) as ws:
            for attempt, expected in (
                (1, "invalid credentials"),
                (2, "invalid credentials"),
                (3, "too many attempts"),
            ):
                send_json(ws, {
                    "type": "auth",
                    "username": "wire_user",
                    "password": "definitely-wrong-%d" % attempt,
                })
                msg = recv_json(ws, 5.0)
                self.assertIsNotNone(msg, "no reply to failed attempt %d" % attempt)
                self.assertEqual(msg.get("type"), "auth_error", msg)
                self.assertEqual(msg.get("reason"), expected, msg)
            # third failure closes with code 1008
            code, reason = recv_close(ws, timeout=5.0)
            self.assertEqual(code, 1008, "close code %r reason %r" % (code, reason))
            self.assertEqual(reason, "too many attempts")

    def test_07_token_auth_on_fresh_connection(self):
        token = type(self).token or self.auth.authenticate("wire_user", GOOD_PASSWORD)
        self.assertTrue(token, "no token available for test_07")
        with connect(self.url, open_timeout=5) as ws:
            send_json(ws, {"type": "auth", "token": token})
            msg = recv_json(ws, 5.0)
            self.assertIsNotNone(msg, "no reply to token auth")
            self.assertEqual(msg.get("type"), "auth_ok", msg)
            self.assertEqual(msg.get("username"), "wire_user", msg)
            self.assertEqual(msg.get("token"), token, msg)
            self._expect_broadcast(ws, "t07")

    def test_08_logout_then_reauth_same_connection(self):
        with connect(self.url, open_timeout=5) as ws:
            msg = self._login(ws)
            token = msg["token"]
            self._expect_broadcast(ws, "t08a")

            send_json(ws, {"type": "logout"})
            reply = recv_json(ws, 3.0)
            self.assertIsNotNone(reply, "no reply to logout")
            self.assertEqual(reply.get("type"), "logged_out", reply)

            # after logout: nothing until re-auth
            self._expect_no_broadcast(ws, wait=1.0)

            # re-auth with the stored token on the SAME connection
            send_json(ws, {"type": "auth", "token": token})
            reply = recv_json(ws, 5.0)
            self.assertIsNotNone(reply, "no reply to re-auth")
            self.assertEqual(reply.get("type"), "auth_ok", reply)
            self.assertEqual(reply.get("username"), "wire_user", reply)

            # broadcasts flow again
            self._expect_broadcast(ws, "t08b")

    def test_09a_bad_messages_return_error(self):
        cases = [
            ("this is not json {", "invalid JSON"),
            ("[1, 2, 3]", "expected a JSON object"),
            ('{"foo": 1}', "missing or invalid message type"),
            ('{"type": ""}', "missing or invalid message type"),
            ('{"type": "ping"}', "unexpected message type"),
            (b"\x01\x02\x03", "binary frames are not supported"),
        ]
        with connect(self.url, open_timeout=5) as ws:
            for raw, reason in cases:
                ws.send(raw)
                msg = recv_json(ws, 3.0)
                self.assertIsNotNone(msg, "no reply to %r" % (raw,))
                self.assertEqual(msg.get("type"), "error", msg)
                self.assertEqual(msg.get("code"), "bad_message", msg)
                self.assertEqual(msg.get("reason"), reason, msg)
            # connection is still usable and still in the auth phase
            send_json(ws, {
                "type": "register",
                "username": "wire_err_user",
                "password": GOOD_PASSWORD,
            })
            msg = recv_json(ws, 3.0)
            self.assertIsNotNone(msg, "connection dead after bad messages")
            self.assertEqual(msg.get("type"), "register_ok", msg)

    def test_09b_silent_connection_closed_on_auth_timeout(self):
        started = time.monotonic()
        with connect(self.url, open_timeout=5) as ws:
            code, reason = recv_close(ws, timeout=13.0)
        elapsed = time.monotonic() - started
        self.assertEqual(
            reason, "authentication timeout",
            "close reason %r after %.1fs" % (reason, elapsed),
        )
        self.assertEqual(code, 1008, "close code %r" % (code,))
        self.assertGreaterEqual(elapsed, 8.0, "closed after only %.1fs" % elapsed)
        self.assertLessEqual(elapsed, 11.5, "closed after %.1fs (expected ~10s)" % elapsed)

    def test_10_broadcast_from_other_thread(self):
        self.assertIsNotNone(self.server.loop)
        self.assertIsNotNone(self.server.thread)
        self.assertNotEqual(self.server.thread, threading.current_thread())
        with connect(self.url, open_timeout=5) as ws:
            self._login(ws)
            errors = []

            def worker():
                try:
                    self.server.broadcast({
                        "type": "telemetry",
                        "raw_ecg": 512, "filtered_ecg": 511.0, "is_r_peak": True,
                        "heart_rate": 72.0, "total_alerts": 0, "total_beats": 9,
                        "timestamp": time.time(),
                    })
                except Exception as exc:  # pragma: no cover - failure path
                    errors.append(exc)

            t = threading.Thread(target=worker)
            t.start()
            t.join(timeout=5)
            self.assertFalse(errors, "broadcast raised: %r" % errors)

            msg = recv_json(ws, 3.0)
            self.assertIsNotNone(msg, "cross-thread broadcast not received")
            self.assertEqual(msg.get("type"), "telemetry")
            self.assertEqual(msg.get("heart_rate"), 72.0)
            self.assertTrue(msg.get("is_r_peak"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
