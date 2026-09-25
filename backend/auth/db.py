"""SQLite-backed user store for LiveGuard dashboard authentication."""

import sqlite3
import threading
from pathlib import Path

from backend import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'viewer',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_login_at TEXT
)
"""


class UserStore:
    """Thread-safe SQLite user store: one shared connection guarded by an RLock.

    Username uniqueness and lookup are case-insensitive; usernames are stored
    exactly as passed. Only hex hashes/salts are stored, never plaintext.
    After close(), further method calls raise RuntimeError.
    """

    def __init__(self, db_path=None):
        if db_path is None:
            db_path = Path(config.DATA_DIR) / "liveguard_users.db"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._closed = False
        with self._lock:
            self._conn.execute(_SCHEMA)
            self._conn.commit()

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("UserStore is closed")

    def create_user(self, username: str, password_hash: str, salt: str, role: str = "viewer") -> bool:
        """Insert a user; False on invalid args or if the username exists."""
        with self._lock:
            self._check_open()
            args = (username, password_hash, salt, role)
            if not all(isinstance(v, str) and v for v in args):
                return False
            try:
                self._conn.execute(
                    "INSERT INTO users (username, password_hash, salt, role) VALUES (?, ?, ?, ?)",
                    args,
                )
                self._conn.commit()
            except sqlite3.IntegrityError:
                self._conn.rollback()
                return False
            return True

    def get_user(self, username: str) -> dict | None:
        """Return {id, username, password_hash, salt, role} or None (case-insensitive)."""
        with self._lock:
            self._check_open()
            row = self._conn.execute(
                "SELECT id, username, password_hash, salt, role FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            return dict(row) if row is not None else None

    def user_count(self) -> int:
        """Number of registered users."""
        with self._lock:
            self._check_open()
            return int(self._conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])

    def record_login(self, username: str) -> None:
        """Stamp last_login_at; silent no-op if the user does not exist."""
        with self._lock:
            self._check_open()
            self._conn.execute(
                "UPDATE users SET last_login_at = datetime('now') WHERE username = ?",
                (username,),
            )
            self._conn.commit()

    def list_users(self) -> list[dict]:
        """Return [{username, role, created_at, last_login_at}, ...] by username.

        Only non-secret columns are selected: password material never leaves
        the store through this method.
        """
        with self._lock:
            self._check_open()
            rows = self._conn.execute(
                "SELECT username, role, created_at, last_login_at "
                "FROM users ORDER BY username"
            ).fetchall()
            return [dict(row) for row in rows]

    def delete_user(self, username: str) -> str:
        """Delete an account: 'deleted', 'not_found', or 'last_user'.

        The existence check, the last-user guard and the DELETE all run under
        this store's lock, so concurrent callers can never race the store down
        to zero accounts (which would lock every operator out).
        """
        with self._lock:
            self._check_open()
            if not isinstance(username, str) or not username:
                return "not_found"
            row = self._conn.execute(
                "SELECT 1 FROM users WHERE username = ?", (username,)
            ).fetchone()
            if row is None:
                return "not_found"
            if self.user_count() <= 1:
                return "last_user"
            self._conn.execute("DELETE FROM users WHERE username = ?", (username,))
            self._conn.commit()
            return "deleted"

    def update_password(self, username: str, password_hash: str, salt: str) -> bool:
        """Set a new password hash + salt; False on invalid args or missing user."""
        with self._lock:
            self._check_open()
            if not all(
                isinstance(v, str) and v for v in (username, password_hash, salt)
            ):
                return False
            cur = self._conn.execute(
                "UPDATE users SET password_hash = ?, salt = ? WHERE username = ?",
                (password_hash, salt, username),
            )
            self._conn.commit()
            return cur.rowcount == 1

    def close(self) -> None:
        """Close the connection; later method calls raise RuntimeError."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self._conn.close()
