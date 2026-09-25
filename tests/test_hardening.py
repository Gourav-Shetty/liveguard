"""Hardening regression tests (stdlib unittest): fail-closed TLS, slow-client
isolation, broadcast fan-out, registration caps, session-message budgets,
map-sweep throttling, server stop(), connection/bind cleanup, startup
readiness (wait_ready) and refund-on-raise for every reserved slot.

Sections:
  A. TLS fail-closed   - LIVEGUARD_TLS_CERT/KEY misconfiguration must never
                         downgrade to plaintext (unit level + start() case)
  B. Slow clients      - send/close timeouts drop exactly one socket, once
  C. Fan-out           - one serialization, per-client isolation, plus an
                         end-to-end broadcast over a real server
  D. Registration caps - per-connection and per-IP success budgets
  E. Session budget    - bad-message cap (1008), stop(), sweep throttling,
                         auth-timeout cleanup, bind failure survivability,
                         wait_ready() startup-readiness signal
  F. Refund-on-raise   - a raising store/verify_token gives back the slot
                         it reserved (password, token and register paths)

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
from websockets.exceptions import ConnectionClosed  # noqa: E402
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


class QueuedWebSocket(FakeWebSocket):
    """Fake socket whose recv() replays a scripted sequence of frames.

    Reading past the end of the script raises immediately instead of
    blocking, so a regression fails fast with a clear message rather than
    tripping the run_bounded() watchdog.
    """

    def __init__(self, messages):
        super().__init__()
        self._inbox = list(messages)

    async def recv(self):
        if not self._inbox:
            raise RuntimeError("recv() called after the scripted messages ran out")
        return self._inbox.pop(0)


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
        # Stop the server FIRST (safe/no-op when it was never started): the
        # serving thread must be gone before the addCleanup rmtree below runs.
        try:
            self.server.stop()
        except (RuntimeError, AttributeError):  # pragma: no cover - defensive
            pass
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


# ======================================================================
# E. Session-phase budget, sweep throttling, stop(), cleanup, bind failure
# ======================================================================
class SessionBudgetTests(_ServerFixture):
    """Authenticated clients cannot earn error replies forever (item 2).

    logout is the ONLY recognized session-phase message and it is terminal
    (it ends the session), so there is no non-terminal recognized message to
    drive an in-session reset on the wire -- the reset is therefore asserted
    at unit level (conn_state counter), the cap end-to-end over a real socket.
    """

    def test_ten_junk_frames_close_session_with_1008(self):
        ok, msg = self.auth.register("session_budget_user", GOOD_PASSWORD)
        self.assertTrue(ok, msg)
        self.server.start()
        wait_until_ready(self.url)
        with connect(self.url, open_timeout=5) as ws:
            send_json(ws, {"type": "auth", "username": "session_budget_user",
                           "password": GOOD_PASSWORD})
            reply = recv_json(ws, timeout=5.0)
            self.assertIsNotNone(reply, "no reply to auth")
            self.assertEqual(reply.get("type"), "auth_ok", reply)
            # 10 well-formed frames of an unrecognized type, back to back.
            for _ in range(10):
                ws.send('{"type": "ping"}')
            replies = []
            code = reason = None
            deadline = time.monotonic() + 5.0
            while True:
                remaining = deadline - time.monotonic()
                self.assertGreater(remaining, 0.0, "connection was not closed")
                try:
                    raw = ws.recv(timeout=remaining)
                except ConnectionClosed as exc:
                    close = exc.rcvd if exc.rcvd is not None else exc.sent
                    if close is not None:
                        code, reason = close.code, close.reason
                    break
                replies.append(json.loads(raw))
            # The first 9 junk frames each earn exactly one error reply; the
            # 10th closes the connection instead of earning another one.
            self.assertEqual(len(replies), 9, replies)
            for reply in replies:
                self.assertEqual(reply.get("type"), "error", reply)
                self.assertEqual(reply.get("code"), "bad_message", reply)
            self.assertEqual(code, 1008, "close code %r reason %r" % (code, reason))
            self.assertEqual(reason, "too many bad messages")

    def test_recognized_message_resets_bad_message_budget(self):
        # Unit level (see class docstring): the counter lives in the
        # per-connection conn_state, so it must survive logout -> re-auth
        # cycles WITHOUT accumulating, while 10 consecutive bad frames in one
        # session still close with 1008.
        junk = '{"type": "ping"}'
        logout = json.dumps({"type": "logout"})
        conn_state = {"register_successes": 0}

        first = QueuedWebSocket([junk] * 9 + [logout])
        result = run_bounded(self.server._session_phase(first, conn_state))
        self.assertEqual(result, "logged_out")
        self.assertEqual(conn_state.get("bad_messages"), 0,
                         "logout must reset the bad-message budget")
        self.assertEqual(first.closes, [])

        # Second session on the SAME connection: 9 more junk frames. Without
        # the reset these would be 18 consecutive and the session would have
        # been closed immediately.
        second = QueuedWebSocket([junk] * 9 + [logout])
        result = run_bounded(self.server._session_phase(second, conn_state))
        self.assertEqual(result, "logged_out",
                         "bad-message budget leaked across sessions")
        self.assertEqual(second.closes, [])

        # ...and the cap itself: 10 consecutive in one session -> 1008.
        third = QueuedWebSocket([junk] * 10 + [logout])
        result = run_bounded(self.server._session_phase(third, conn_state))
        self.assertEqual(result, "closed")
        self.assertEqual(third.closes, [(1008, "too many bad messages")])
        self.assertEqual(conn_state.get("bad_messages"),
                         ts_module.MAX_SESSION_BAD_MESSAGES)


class SweepThrottleTests(_ServerFixture):
    """Full-map expiry sweeps are throttled; per-IP windows stay exact (6)."""

    def test_attempt_sweep_throttled_but_queried_ip_expires_exactly(self):
        with mock.patch.object(ts_module, "MAX_IP_AUTH_FAILURES", 2), \
                mock.patch.object(ts_module, "IP_FAILURE_WINDOW_SECONDS", 0.05):
            self.assertTrue(self.server._reserve_ip_attempt("8.8.8.8"))
            self.assertTrue(self.server._reserve_ip_attempt("8.8.8.8"))
            self.assertFalse(self.server._reserve_ip_attempt("8.8.8.8"))
            time.sleep(0.15)  # 8.8.8.8's window elapses
            # Reserving for a DIFFERENT ip must not sweep the whole map...
            self.assertTrue(self.server._reserve_ip_attempt("4.4.4.4"))
            self.assertIn(
                "8.8.8.8", self.server._ip_failures,
                "full-map sweep must be throttled to at most once per %.0fs"
                % ts_module.IP_SWEEP_INTERVAL_SECONDS,
            )
            # ...but the queried ip's own window is expiry-filtered exactly.
            self.assertTrue(self.server._reserve_ip_attempt("8.8.8.8"))

    def test_registration_sweep_throttled_but_queried_ip_expires_exactly(self):
        with mock.patch.object(ts_module, "MAX_IP_REGISTER_SUCCESSES", 2), \
                mock.patch.object(ts_module, "IP_REGISTER_WINDOW_SECONDS", 0.05):
            self.assertTrue(self.server._reserve_ip_registration("9.9.9.1"))
            self.assertTrue(self.server._reserve_ip_registration("9.9.9.1"))
            self.assertFalse(self.server._reserve_ip_registration("9.9.9.1"))
            time.sleep(0.15)  # 9.9.9.1's window elapses
            self.assertTrue(self.server._reserve_ip_registration("9.9.9.2"))
            self.assertIn(
                "9.9.9.1", self.server._ip_reg_successes,
                "full-map sweep must be throttled to at most once per %.0fs"
                % ts_module.IP_SWEEP_INTERVAL_SECONDS,
            )
            self.assertTrue(self.server._reserve_ip_registration("9.9.9.1"))


class ServerStopTests(_ServerFixture):
    """stop(): joins the serving thread, frees the port, idempotent (7)."""

    def test_stop_is_safe_when_never_started(self):
        self.server.stop()  # no thread, no future -> no-op
        self.server.stop()  # idempotent
        self.assertIsNone(self.server.thread)

    def test_stop_joins_thread_and_releases_port(self):
        ok, msg = self.auth.register("stop_user", GOOD_PASSWORD)
        self.assertTrue(ok, msg)
        self.server.start()
        wait_until_ready(self.url)
        with connect(self.url, open_timeout=5) as ws:
            send_json(ws, {"type": "auth", "username": "stop_user",
                           "password": GOOD_PASSWORD})
            reply = recv_json(ws, timeout=5.0)
            self.assertIsNotNone(reply, "no reply to auth")
            self.assertEqual(reply.get("type"), "auth_ok", reply)

            # Stop WITH a live authenticated connection: the close handshake
            # runs on websockets' side, the handler must still finish.
            self.server.stop()
            self.assertIsNotNone(self.server.thread)
            self.assertFalse(self.server.thread.is_alive(),
                             "stop() must join the serving thread")

        # No listener remains: a fresh connect is refused...
        with self.assertRaises(OSError):
            with connect(self.url, open_timeout=2):
                pass
        # ...and the port can be bound again. SO_REUSEADDR only excuses
        # TIME_WAIT sockets left by the connections above; on Windows an
        # actively listening socket would still block this bind.
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", self.server.port))
        finally:
            probe.close()
        # stop() again: idempotent after a full shutdown.
        self.server.stop()
        self.assertFalse(self.server.thread.is_alive())


class ConnectionCleanupTests(_ServerFixture):
    """Auth timeouts must release every per-connection entry (8c)."""

    def _wait_for(self, predicate, what: str, timeout: float = 3.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.05)
        self.fail("%s did not happen within %.1fs" % (what, timeout))

    def test_auth_timeout_cleans_up_after_three_cycles(self):
        ok, msg = self.auth.register("cleanup_user", GOOD_PASSWORD)
        self.assertTrue(ok, msg)
        self.server.start()
        wait_until_ready(self.url)

        # Sanity: an authenticated connection IS tracked while open...
        with connect(self.url, open_timeout=5) as ws:
            send_json(ws, {"type": "auth", "username": "cleanup_user",
                           "password": GOOD_PASSWORD})
            reply = recv_json(ws, timeout=5.0)
            self.assertIsNotNone(reply, "no reply to auth")
            self.assertEqual(reply.get("type"), "auth_ok", reply)
            self.assertEqual(len(self.server.connected_clients), 1)
        self._wait_for(lambda: not self.server.connected_clients,
                       "authed client cleanup")

        # ...then three connect-without-auth cycles. AUTH_TIMEOUT_SECONDS is
        # read at _auth_phase entry, so the patch applies to every connection
        # opened inside this block (proven by the fast close below).
        with mock.patch.object(ts_module, "AUTH_TIMEOUT_SECONDS", 0.3):
            for cycle in range(3):
                with connect(self.url, open_timeout=5) as ws:
                    started = time.monotonic()
                    code, reason = recv_close(ws, timeout=3.0)
                    elapsed = time.monotonic() - started
                    self.assertEqual(code, 1008, "cycle %d: %r" % (cycle, reason))
                    self.assertEqual(reason, "authentication timeout")
                    self.assertLess(elapsed, 2.0,
                                    "patched auth timeout not in effect (%.2fs)"
                                    % elapsed)

        def settled():
            return (not self.server.connected_clients
                    and not self.server._slow_clients)

        self._wait_for(settled, "handler cleanup after auth timeouts")
        self.assertEqual(set(self.server.connected_clients), set())
        self.assertEqual(set(self.server._slow_clients), set())


class BindFailureTests(_ServerFixture):
    """A taken port must fail cleanly and stop() stay safe (8d).

    Also asserts the startup-readiness signal stays dark on that failure,
    which is what run_edge uses to notice that no telemetry can flow.
    """

    def test_start_on_taken_port_fails_and_stop_is_safe(self):
        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                # Windows: without this, an SO_REUSEADDR bind elsewhere could
                # squat on the same port. Set BEFORE bind, as documented.
                blocker.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            blocker.bind(("127.0.0.1", 0))  # bound but NOT listening
            port = blocker.getsockname()[1]

            server = TelemetryServer(host="127.0.0.1", port=port, auth=self.auth)
            # Capture the expected "telemetry server failed" log so the
            # deliberate bind failure does not spam the suite output -- and
            # assert the failure was actually reported, not swallowed.
            with self.assertLogs("backend.telemetry_server", level="ERROR") as logs:
                server.start()
                # The bind fails inside _serve, so the thread must die on its own.
                deadline = time.monotonic() + 3.0
                while server.thread.is_alive() and time.monotonic() < deadline:
                    time.sleep(0.05)
            self.assertFalse(server.thread.is_alive(),
                             "serving thread survived a bind failure")
            self.assertTrue(
                any("telemetry server failed" in line for line in logs.output),
                logs.output,
            )
            # No listener: a websocket connect must fail (refused -> OSError).
            with self.assertRaises(OSError):
                with connect("ws://127.0.0.1:%d" % port, open_timeout=2):
                    pass
            # stop() after the fact is safe and idempotent.
            server.stop()
            self.assertFalse(server.thread.is_alive())
        finally:
            blocker.close()

    def test_wait_ready_false_after_bind_failure(self):
        # A bind failure is only ever logged inside the serving thread; the
        # readiness signal must therefore stay dark so a caller (run_edge)
        # can tell that NO telemetry will be delivered.
        blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                blocker.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            blocker.bind(("127.0.0.1", 0))  # bound but NOT listening
            port = blocker.getsockname()[1]

            server = TelemetryServer(host="127.0.0.1", port=port, auth=self.auth)
            with self.assertLogs("backend.telemetry_server", level="ERROR"):
                server.start()
                deadline = time.monotonic() + 3.0
                while server.thread.is_alive() and time.monotonic() < deadline:
                    time.sleep(0.05)
            self.assertFalse(server.thread.is_alive(),
                             "serving thread survived a bind failure")
            # Never set by the failed bind: wait_ready() must report False
            # once its timeout elapses (it waits the full window, it does
            # not pretend to be ready)...
            started = time.monotonic()
            self.assertFalse(server.wait_ready(0.3))
            self.assertGreaterEqual(
                time.monotonic() - started, 0.25,
                "wait_ready() returned before its timeout without being ready",
            )
            # ...and stop() keeps readiness cleared.
            server.stop()
            self.assertFalse(server.wait_ready(0.3))
        finally:
            blocker.close()


class StartupReadinessTests(_ServerFixture):
    """wait_ready(): True only while the server is actually serving (8e).

    Covers the signal run_edge relies on to notice a silent startup
    failure, plus the start()/stop() hygiene that keeps it truthful
    across a stop.
    """

    def test_wait_ready_true_after_start_then_cleared_by_stop(self):
        # Never started -> not ready (and it must stay False, not flicker).
        self.assertFalse(self.server.wait_ready(0.2))

        self.server.start()
        started = time.monotonic()
        self.assertTrue(
            self.server.wait_ready(5.0),
            "wait_ready() never reported a successfully bound server",
        )
        self.assertLess(
            time.monotonic() - started, 3.0,
            "wait_ready() should fire as soon as the listener is bound",
        )
        # The signal is honest: a client can actually connect right now.
        with connect(self.url, open_timeout=5):
            pass

        self.server.stop()
        self.assertFalse(
            self.server.wait_ready(0.2),
            "stop() must clear readiness",
        )


# ======================================================================
# F. Refund-on-raise: a broken store must not leak rate-limit budget
# ======================================================================
class RefundOnRaiseTests(_ServerFixture):
    """Reserve-before-attempt slots come back when the store RAISES.

    The register/authenticate calls run in asyncio.to_thread and can raise
    (sqlite3.OperationalError, disk full, ...), and verify_token runs inline
    in _auth_phase and can raise just the same. Without the refund the
    per-IP caps would silently shrink for the rest of the window on every
    such incident; these tests force the exception and assert the slot
    returns while the connection still dies via the handler catch-all.
    """

    def test_register_raise_refunds_ip_registration_slot(self):
        self.server.start()
        wait_until_ready(self.url)
        with mock.patch.object(ts_module, "MAX_IP_REGISTER_SUCCESSES", 2):
            # Occupy slot 1 with a real success.
            with connect(self.url, open_timeout=5) as conn1:
                send_json(conn1, {"type": "register", "username": "refund_ok",
                                  "password": GOOD_PASSWORD})
                reply = recv_json(conn1, timeout=5.0)
                self.assertEqual((reply or {}).get("type"), "register_ok", reply)
            self.assertEqual(
                len(self.server._ip_reg_successes.get("127.0.0.1", [])), 1)

            # This attempt reserves slot 2, then the store raises: the slot
            # must come back and the handler catch-all must kill the socket.
            with mock.patch.object(self.auth, "register",
                                   side_effect=RuntimeError("boom")):
                with connect(self.url, open_timeout=5) as conn2:
                    send_json(conn2, {"type": "register",
                                      "username": "refund_boom",
                                      "password": GOOD_PASSWORD})
                    recv_close(conn2, timeout=5.0)

            # Refunded: still exactly one stamp, no account was created.
            self.assertEqual(
                len(self.server._ip_reg_successes.get("127.0.0.1", [])), 1,
                "raised register attempt leaked its IP reservation",
            )
            self.assertIsNone(self.store.get_user("refund_boom"))
            # ...and the freed slot is genuinely usable again (cap = 2).
            with connect(self.url, open_timeout=5) as conn3:
                send_json(conn3, {"type": "register", "username": "refund_later",
                                  "password": GOOD_PASSWORD})
                reply = recv_json(conn3, timeout=5.0)
                self.assertEqual((reply or {}).get("type"), "register_ok", reply)
            self.assertEqual(
                len(self.server._ip_reg_successes.get("127.0.0.1", [])), 2)

    def test_authenticate_raise_refunds_attempt_slot(self):
        self.server.start()
        wait_until_ready(self.url)
        ok, msg = self.auth.register("refund_user", GOOD_PASSWORD)
        self.assertTrue(ok, msg)
        # Fresh budget: no failed attempts recorded for this IP yet.
        self.assertIsNone(self.server._ip_failures.get("127.0.0.1"))

        with mock.patch.object(self.auth, "authenticate",
                               side_effect=RuntimeError("boom")):
            with connect(self.url, open_timeout=5) as conn:
                send_json(conn, {"type": "auth", "username": "refund_user",
                                 "password": GOOD_PASSWORD})
                recv_close(conn, timeout=5.0)

        # The reserved attempt slot was refunded all the way back to empty:
        # _refund_ip_attempt pops the key when the count reaches zero.
        self.assertIsNone(
            self.server._ip_failures.get("127.0.0.1"),
            "raised authenticate leaked its IP attempt reservation",
        )

    def test_verify_token_raise_refunds_attempt_slot(self):
        # Same class of bug as the password branch: the token branch reserves
        # the per-IP slot BEFORE verify_token runs, so a raising verify_token
        # must give it back instead of silently shrinking the budget.
        self.server.start()
        wait_until_ready(self.url)
        # Fresh budget: no failed attempts recorded for this IP yet.
        self.assertIsNone(self.server._ip_failures.get("127.0.0.1"))

        with mock.patch.object(self.auth, "verify_token",
                               side_effect=RuntimeError("boom")):
            with connect(self.url, open_timeout=5) as conn:
                # Token auth frame: {"type": "auth", "token": "..."} -- a
                # string token is what reaches verify_token in _auth_phase.
                send_json(conn, {"type": "auth", "token": "bogus-token"})
                # The re-raised exception dies in _handler's catch-all, so
                # the connection is closed rather than answered.
                recv_close(conn, timeout=5.0)

        # The reserved attempt slot was refunded all the way back to empty:
        # _refund_ip_attempt pops the key when the count reaches zero.
        self.assertIsNone(
            self.server._ip_failures.get("127.0.0.1"),
            "raised verify_token leaked its IP attempt reservation",
        )


if __name__ == "__main__":
    unittest.main()
