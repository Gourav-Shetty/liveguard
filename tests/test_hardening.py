"""Hardening regression tests (stdlib unittest): fail-closed TLS, slow-client
isolation, broadcast fan-out, and registration caps.

Sections:
  A. TLS fail-closed   - LIVEGUARD_TLS_CERT/KEY misconfiguration must never
                         downgrade to plaintext (unit level + start() case)
  B. Slow clients      - send/close timeouts drop exactly one socket, once
  C. Fan-out           - one serialization, per-client isolation, plus an
                         end-to-end broadcast over a real server
  D. Registration caps - per-connection and per-IP success budgets

LIVEGUARD_DATA_DIR is pointed at a throwaway temp directory BEFORE
backend.config is imported (only when no earlier test module already set it),
so the repo's data/ directory is never read or written by these tests, and
tearDownModule only ever removes a directory this module created. Every
TelemetryServer is built with an explicit AuthService backed by a temp-dir
UserStore for the same reason.

Run from the repo root:
    python -m unittest tests.test_hardening -v
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Flag BEFORE setdefault: an earlier test module (test_auth_flow) may have
# already pointed LIVEGUARD_DATA_DIR at its own temp dir -- a directory this
# module must never delete. Only remove a dir we created ourselves.
CREATED_DATA_DIR = "LIVEGUARD_DATA_DIR" not in os.environ
if CREATED_DATA_DIR:
    TMP_DATA_DIR = tempfile.mkdtemp(prefix="liveguard_hardening_tests_")
else:
    TMP_DATA_DIR = os.environ["LIVEGUARD_DATA_DIR"]
os.environ.setdefault("LIVEGUARD_DATA_DIR", TMP_DATA_DIR)  # BEFORE backend.config import
# Registration policy: the wire tests below exercise registration mechanics
# (multiple accounts per store), so keep registration open module-wide.
os.environ.setdefault("LIVEGUARD_ALLOW_REGISTER", "1")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend import config  # noqa: E402
from backend import telemetry_server as ts_module  # noqa: E402
from backend.auth.db import UserStore  # noqa: E402
from backend.auth.service import AuthService  # noqa: E402
from backend.telemetry_server import TelemetryServer  # noqa: E402
from websockets.sync.client import connect  # noqa: E402

GOOD_PASSWORD = "s3cret-pass"  # 11 chars -> satisfies the 8-128 rule
TLS_ENV_KEYS = ("LIVEGUARD_TLS_CERT", "LIVEGUARD_TLS_KEY")


def tearDownModule():
    if CREATED_DATA_DIR:
        shutil.rmtree(TMP_DATA_DIR, ignore_errors=True)


# ----------------------------------------------------------------------
# Helpers (self-contained copies of the test_auth_flow utilities)
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


def recv_json(ws, timeout: float = 5.0):
    """Next frame as a dict, or None if nothing arrived before the timeout."""
    try:
        raw = ws.recv(timeout=timeout)
    except TimeoutError:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    return json.loads(raw)


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


def run_bounded(coro, timeout: float = 15.0):
    """``asyncio.run`` with a watchdog.

    The slow-client tests park a fake socket forever on purpose; if the
    production send/close timeout ever regresses, the test must FAIL fast
    instead of hanging the suite (and the CI job) until its external
    timeout kills it with no diagnostic.
    """
    async def _watchdog():
        return await asyncio.wait_for(coro, timeout)

    return asyncio.run(_watchdog())


def pop_tls_env() -> dict:
    """Remove both TLS env vars, returning them for restore_tls_env()..

    Done with explicit pop/finally (never patch.dict(clear=True)) so the
    environment of other test modules is never wholesale-clobbered.
    """
    return {key: os.environ.pop(key, None) for key in TLS_ENV_KEYS}


def restore_tls_env(saved: dict) -> None:
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


# ----------------------------------------------------------------------
# Fake sockets for unit-level slow-client / fan-out tests (no network)
# ----------------------------------------------------------------------
class FakeTransport:
    """Records abort() calls the slow-client drop performs on TCP."""

    def __init__(self):
        self.aborts = 0

    def abort(self):
        self.aborts += 1


class FakeWebSocket:
    """Minimal stand-in for a server-side websockets connection.

    ``hang_send``/``hang_close`` park the coroutine forever so the server's
    send/close timeouts fire; otherwise calls are recorded and return.
    """

    def __init__(self, hang_send: bool = False, hang_close: bool = False,
                 transport: FakeTransport | None = None):
        self.remote_address = ("10.0.0.9", 40000)
        self.sent = []
        self.sent_at = []  # monotonic timestamps, one per delivered message
        self.closes = []  # list of (code, reason)
        self._hang_send = hang_send
        self._hang_close = hang_close
        if transport is not None:
            self.transport = transport

    async def send(self, message):
        if self._hang_send:
            await asyncio.Event().wait()  # never completes -> send timeout
        self.sent.append(message)
        self.sent_at.append(time.monotonic())

    async def close(self, code=None, reason=None):
        if self._hang_close:
            await asyncio.Event().wait()  # never completes -> close timeout
        self.closes.append((code, reason))


# ----------------------------------------------------------------------
# Shared fixture: one TelemetryServer + throwaway store per test
# ----------------------------------------------------------------------
class _ServerFixture(unittest.TestCase):
    """Per-test TelemetryServer wired to a temp-dir UserStore.

    Never constructs a default AuthService, so nothing can touch the repo's
    data/ directory. No test here shares mutable state with any other.
    """

    def setUp(self):
        # Register cleanups FIRST: unittest skips tearDown when setUp
        # raises, so env/directory restoration must not depend on it.
        saved_allow = os.environ.get("LIVEGUARD_ALLOW_REGISTER")

        def _restore_allow_register():
            if saved_allow is None:
                os.environ.pop("LIVEGUARD_ALLOW_REGISTER", None)
            else:
                os.environ["LIVEGUARD_ALLOW_REGISTER"] = saved_allow

        self.addCleanup(_restore_allow_register)
        os.environ["LIVEGUARD_ALLOW_REGISTER"] = "1"

        self.dir = tempfile.mkdtemp(prefix="lg_hardening_")
        self.addCleanup(shutil.rmtree, self.dir, True)  # runs after tearDown

        # Hermetic regardless of module import order: pin DATA_DIR to this
        # test's own directory so no code path can read another module's
        # (possibly already deleted) data directory. Cleanups stop the
        # patcher after tearDown.
        data_dir_patcher = mock.patch.object(config, "DATA_DIR", self.dir)
        data_dir_patcher.start()
        self.addCleanup(data_dir_patcher.stop)

        self.store = UserStore(db_path=os.path.join(self.dir, "users.db"))
        self.auth = AuthService(store=self.store)
        self.server = TelemetryServer(
            host="127.0.0.1", port=free_port(), auth=self.auth
        )
        self.url = "ws://127.0.0.1:%d" % self.server.port

    def tearDown(self):
        # Close the sqlite handle before the addCleanup rmtree below runs
        # (Windows cannot delete an open database file).
        try:
            self.store.close()
        except (RuntimeError, AttributeError):
            pass


# ======================================================================
# A. TLS fail-closed
# ======================================================================
class TLSFailClosedTests(_ServerFixture):
    """TLS misconfiguration must fail closed, never fall back to ws://."""

    def test_both_env_unset_builds_no_context(self):
        saved = pop_tls_env()
        try:
            self.assertIsNone(TelemetryServer._build_ssl_context())
        finally:
            restore_tls_env(saved)

    def test_cert_only_raises_value_error(self):
        saved = pop_tls_env()
        try:
            os.environ["LIVEGUARD_TLS_CERT"] = os.path.join(self.dir, "cert.pem")
            with self.assertRaises(ValueError):
                TelemetryServer._build_ssl_context()
        finally:
            restore_tls_env(saved)

    def test_key_only_raises_value_error(self):
        saved = pop_tls_env()
        try:
            os.environ["LIVEGUARD_TLS_KEY"] = os.path.join(self.dir, "key.pem")
            with self.assertRaises(ValueError):
                TelemetryServer._build_ssl_context()
        finally:
            restore_tls_env(saved)

    def test_missing_pem_files_raise_os_error(self):
        # ssl.SSLError subclasses OSError, so OSError covers cert/key load errors.
        saved = pop_tls_env()
        try:
            os.environ["LIVEGUARD_TLS_CERT"] = os.path.join(self.dir, "no_such_cert.pem")
            os.environ["LIVEGUARD_TLS_KEY"] = os.path.join(self.dir, "no_such_key.pem")
            with self.assertRaises(OSError):
                TelemetryServer._build_ssl_context()
        finally:
            restore_tls_env(saved)

    def test_start_without_valid_tls_never_starts_thread(self):
        # Requested-but-broken TLS: log the error and refuse to serve rather
        # than downgrade to plaintext. No thread, no event loop, no bind.
        saved = pop_tls_env()
        try:
            os.environ["LIVEGUARD_TLS_CERT"] = os.path.join(self.dir, "missing_cert.pem")
            os.environ["LIVEGUARD_TLS_KEY"] = os.path.join(self.dir, "missing_key.pem")
            self.server.start()
            self.assertIsNone(
                self.server.thread, "start() must not launch a thread when TLS setup fails"
            )
            self.assertIsNone(
                self.server.loop, "no event loop may run when TLS setup fails"
            )
        finally:
            restore_tls_env(saved)


# ======================================================================
# B. Slow-client drop
# ======================================================================
class SlowClientDropTests(_ServerFixture):
    """Unit level: _drop_slow_client / _send with fake sockets (no network)."""

    def test_drop_is_idempotent(self):
        transport = FakeTransport()
        fake = FakeWebSocket(transport=transport)
        self.server.connected_clients.add(fake)
        run_bounded(self.server._drop_slow_client(fake, "broadcast"))
        run_bounded(self.server._drop_slow_client(fake, "broadcast"))
        # Exactly ONE close, with the policy-violation code and reason...
        self.assertEqual(fake.closes, [(1008, "slow client")])
        # ...the socket is out of the broadcast set and remembered as slow...
        self.assertNotIn(fake, self.server.connected_clients)
        self.assertIn(fake, self.server._slow_clients)
        # ...and because the closing handshake completed, TCP was NOT aborted.
        self.assertEqual(transport.aborts, 0)

    def test_send_timeout_drops_client(self):
        fake = FakeWebSocket(hang_send=True)
        self.server.connected_clients.add(fake)
        with mock.patch.object(ts_module, "CLIENT_SEND_TIMEOUT", 0.15):
            run_bounded(self.server._send(fake, {"type": "ping"}))
        self.assertEqual(fake.closes, [(1008, "slow client")])
        self.assertEqual(fake.sent, [], "hung send must not have landed")
        self.assertNotIn(fake, self.server.connected_clients)
        self.assertIn(fake, self.server._slow_clients)

    def test_close_handshake_timeout_aborts_transport(self):
        transport = FakeTransport()
        fake = FakeWebSocket(hang_close=True, transport=transport)
        self.server.connected_clients.add(fake)
        with mock.patch.object(ts_module, "SLOW_CLIENT_CLOSE_TIMEOUT", 0.05):
            run_bounded(self.server._drop_slow_client(fake, "broadcast"))
        # The closing handshake never finished -> the TCP connection is aborted.
        self.assertEqual(transport.aborts, 1)
        self.assertEqual(fake.closes, [], "close must never have completed")
        self.assertNotIn(fake, self.server.connected_clients)
        self.assertIn(fake, self.server._slow_clients)


# ======================================================================
# C. Fan-out / broadcast
# ======================================================================
class FanoutTests(_ServerFixture):
    """One serialized message, per-client isolation, end-to-end broadcast."""

    def test_fanout_isolates_slow_client_and_serializes_once(self):
        ok1 = FakeWebSocket()
        ok2 = FakeWebSocket()
        slow = FakeWebSocket(hang_send=True)
        for ws in (ok1, ok2, slow):
            self.server.connected_clients.add(ws)
        message = json.dumps({"type": "beat", "marker": "fanout_iso"})
        started = time.monotonic()
        # 1.0s send timeout: a *sequential* fan-out would deliver the healthy
        # clients only AFTER the slow one times out (~1.0s); concurrent
        # delivery gets it to them almost immediately.
        with mock.patch.object(ts_module, "CLIENT_SEND_TIMEOUT", 1.0):
            run_bounded(self.server._fanout(message, [slow, ok1, ok2]),
                        timeout=20.0)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 5.0, "fan-out never returned")
        # Healthy clients each got exactly this message...
        self.assertEqual(ok1.sent, [message])
        self.assertEqual(ok2.sent, [message])
        # ...BEFORE the slow client's 1.0s timeout expired: proof they were
        # not queued behind it.
        for ok in (ok1, ok2):
            self.assertLess(
                ok.sent_at[0] - started, 0.5,
                "healthy client waited on the slow client's timeout",
            )
        # ...and it is the SAME string object: serialized exactly once.
        self.assertIs(
            ok1.sent[0], ok2.sent[0], "message must be serialized once and shared"
        )
        # The slow client was dropped; the healthy ones were not.
        self.assertNotIn(slow, self.server.connected_clients)
        self.assertIn(ok1, self.server.connected_clients)
        self.assertIn(ok2, self.server.connected_clients)
        self.assertIn(slow, self.server._slow_clients)

    def test_broadcast_without_loop_is_noop(self):
        self.assertIsNone(self.server.loop, "fixture server must be unstarted")
        # loop is None -> broadcast must return silently, not raise.
        self.server.broadcast({"type": "x", "marker": "never_sent"})

    def test_broadcast_reaches_all_authenticated_clients(self):
        # setUp already pins config.DATA_DIR to this test's own directory,
        # so token-secret generation stays hermetic regardless of which
        # module set LIVEGUARD_DATA_DIR at import time.
        ok, msg = self.auth.register("fanout_user", GOOD_PASSWORD)
        self.assertTrue(ok, msg)
        self.server.start()
        wait_until_ready(self.url)
        with connect(self.url, open_timeout=5) as ws1, \
                connect(self.url, open_timeout=5) as ws2:
            for ws in (ws1, ws2):
                send_json(ws, {"type": "auth", "username": "fanout_user",
                               "password": GOOD_PASSWORD})
                reply = recv_json(ws, timeout=5.0)
                self.assertIsNotNone(reply, "no reply to password auth")
                self.assertEqual(reply.get("type"), "auth_ok", reply)
                self.assertEqual(reply.get("username"), "fanout_user", reply)
            payload = {
                "type": "beat", "marker": "fanout_probe",
                "prediction": "Normal", "is_anomaly": False,
                "confidence": 98.5, "timestamp": time.time(),
            }
            self.server.broadcast(payload)
            got1 = recv_json(ws1, timeout=5.0)
            got2 = recv_json(ws2, timeout=5.0)
            self.assertIsNotNone(got1, "client 1 received no broadcast")
            self.assertIsNotNone(got2, "client 2 received no broadcast")
            self.assertEqual(got1, payload)
            self.assertEqual(got2, payload)
            self.assertEqual(got1, got2)


# ======================================================================
# D. Registration caps
# ======================================================================
class RegisterCapTests(_ServerFixture):
    """Per-connection and per-IP registration-success budgets.

    Wire tests start their own server so each gets a fresh per-IP budget;
    the unit tests use the fresh unstarted server from setUp directly.
    """

    # -- helpers ------------------------------------------------------
    def _register(self, ws, username: str, timeout: float = 5.0) -> dict:
        send_json(ws, {"type": "register", "username": username,
                       "password": GOOD_PASSWORD})
        reply = recv_json(ws, timeout=timeout)
        self.assertIsNotNone(reply, "no reply to register %r" % username)
        return reply

    def _assert_register_ok(self, reply: dict, username: str) -> None:
        self.assertEqual(reply.get("type"), "register_ok", reply)
        self.assertEqual(reply.get("username"), username, reply)

    def _assert_register_denied(self, reply: dict, reason: str) -> None:
        self.assertEqual(reply.get("type"), "error", reply)
        self.assertEqual(reply.get("code"), "register_failed", reply)
        self.assertEqual(reply.get("reason"), reason, reply)

    # -- wire ---------------------------------------------------------
    def test_per_connection_cap_denies_second_success(self):
        self.server.start()
        wait_until_ready(self.url)
        with connect(self.url, open_timeout=5) as conn1:
            reply = self._register(conn1, "cap_user_a")
            self._assert_register_ok(reply, "cap_user_a")
            # Second success on the SAME socket -> per-connection cap.
            reply = self._register(conn1, "cap_user_b")
            self._assert_register_denied(reply, "registration limit reached")
            # Still denied for yet another user (cap is on successes, not names).
            reply = self._register(conn1, "cap_user_c")
            self._assert_register_denied(reply, "registration limit reached")
            # A denied attempt must not have created anything.
            self.assertIsNone(self.store.get_user("cap_user_b"))
            self.assertIsNone(self.store.get_user("cap_user_c"))
        # A fresh socket gets its own per-connection budget...
        with connect(self.url, open_timeout=5) as conn2:
            reply = self._register(conn2, "cap_user_b")
            self._assert_register_ok(reply, "cap_user_b")
        # ...so cap_user_b now exists, while the denied cap_user_c never did.
        self.assertIsNotNone(self.store.get_user("cap_user_b"))
        self.assertIsNone(self.store.get_user("cap_user_c"))
        self.assertIsNotNone(self.store.get_user("cap_user_a"))

    def test_per_ip_cap_denies_other_connections(self):
        self.server.start()
        wait_until_ready(self.url)
        with mock.patch.object(ts_module, "MAX_IP_REGISTER_SUCCESSES", 1):
            with connect(self.url, open_timeout=5) as conn1:
                reply = self._register(conn1, "per_ip_user_a")
                self._assert_register_ok(reply, "per_ip_user_a")
            # A second connection from the same IP is over the per-IP budget.
            with connect(self.url, open_timeout=5) as conn2:
                reply = self._register(conn2, "per_ip_user_b")
                self._assert_register_denied(reply, "too many registrations")
            # A denied attempt must not have created anything.
            self.assertIsNone(self.store.get_user("per_ip_user_b"))
            # Budget is exhausted... a reserve is refused...
            self.assertFalse(self.server._reserve_ip_registration("127.0.0.1"))
            # ...until a refund restores the slot.
            self.server._refund_ip_registration("127.0.0.1")
            self.assertTrue(self.server._reserve_ip_registration("127.0.0.1"))

    # -- unit ---------------------------------------------------------
    def test_ip_registration_window_expires(self):
        with mock.patch.object(ts_module, "MAX_IP_REGISTER_SUCCESSES", 2), \
                mock.patch.object(ts_module, "IP_REGISTER_WINDOW_SECONDS", 0.05):
            ip = "1.2.3.4"
            self.assertTrue(self.server._reserve_ip_registration(ip))
            self.assertTrue(self.server._reserve_ip_registration(ip))
            self.assertFalse(self.server._reserve_ip_registration(ip))
            time.sleep(0.15)  # window (0.05s) elapses; next reserve sweeps it
            self.assertTrue(self.server._reserve_ip_registration(ip))

    def test_ip_registration_burst_cannot_exceed_cap(self):
        results = []

        def worker():
            try:
                results.append(self.server._reserve_ip_registration("1.2.3.4"))
            except Exception as exc:  # pragma: no cover - diagnostics
                results.append("EXC:%s" % exc)

        with mock.patch.object(ts_module, "MAX_IP_REGISTER_SUCCESSES", 5):
            threads = [threading.Thread(target=worker) for _ in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=15)

        granted = [r for r in results if r is True]
        denied = [r for r in results if r is False]
        errors = [r for r in results if isinstance(r, str)]
        # Reserve-before-create: exactly MAX slots, extras denied, never over.
        self.assertEqual(len(results), 20, results)
        self.assertFalse(errors, errors)
        self.assertEqual(len(granted), 5, results)
        self.assertEqual(len(denied), 15, results)


if __name__ == "__main__":
    unittest.main()
