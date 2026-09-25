import asyncio
import json
import logging
import os
import ssl
import threading
import time

import websockets
from websockets.exceptions import ConnectionClosed

from backend import config
from backend.auth import AuthService

logger = logging.getLogger(__name__)
# Give warnings/errors a home on stderr, but only while nothing else owns the
# root logger: an embedding application (or the test suite) may already have
# installed its own handlers and those must never be clobbered.
if not logging.getLogger().handlers:
    logging.basicConfig()

# A newly connected client must complete authentication within this window.
AUTH_TIMEOUT_SECONDS = 10.0
# Failed authentication attempts allowed per connection before disconnecting.
MAX_AUTH_FAILURES = 3
# Failed registrations allowed per connection before disconnecting.
MAX_REGISTER_FAILURES = 5
# Failed-auth attempts allowed per client IP within the window. The slot is
# reserved BEFORE the password is checked and refunded on success, so the cap
# is exact even against parallel connection bursts, and it tracks *failures*
# (a correct login never consumes budget). Keyed by IP -- never by username --
# so an unauthenticated attacker cannot lock a specific account out.
MAX_IP_AUTH_FAILURES = 10
IP_FAILURE_WINDOW_SECONDS = 300.0
# Successful *registrations* allowed per connection: the viewer's flow is
# register -> auto sign-in, so one success per socket is plenty. Like the
# failed-auth budget this is a reserve-before-attempt cap, so the per-IP
# limit is never overshot by a burst of parallel connections.
MAX_REGISTER_SUCCESSES_PER_CONNECTION = 1
# Successful registrations allowed per client IP within the rolling window.
# Generous on purpose: it only slows bulk account creation from one address,
# it is not meant to block legitimate shared-NAT deployments.
MAX_IP_REGISTER_SUCCESSES = 25
IP_REGISTER_WINDOW_SECONDS = 600.0
# Full-map expiry sweeps of the per-IP budgets run at most this often, so a
# burst of reserves costs O(1) instead of O(tracked keys) while holding the
# lock. The entry for the IP actually being reserved is ALWAYS expiry-filtered
# (see _reserve_ip_*), so per-IP window semantics stay exact between sweeps.
IP_SWEEP_INTERVAL_SECONDS = 5.0
# Consecutive malformed/unknown messages an authenticated client may send in
# the session phase before it is disconnected with 1008. The counter lives in
# the per-connection state and is reset whenever a recognized message (logout)
# arrives, so a briefly glitching client is not cut off but a frame-spammer
# can no longer earn error replies forever.
MAX_SESSION_BAD_MESSAGES = 10
# Every send (broadcast or direct reply) is best-effort and bounded: a client
# that cannot absorb a message within this window is treated as slow and
# disconnected instead of being buffered to forever.
CLIENT_SEND_TIMEOUT = 5.0
# Grace period for the closing handshake with a slow client before its TCP
# connection is aborted outright.
SLOW_CLIENT_CLOSE_TIMEOUT = 2.0
# Delay inserted before every failed-auth reply (rate limiting).
AUTH_FAILURE_DELAY = 0.3
# WebSocket close status code used for policy violations (auth failures).
POLICY_VIOLATION_CLOSE_CODE = 1008


class TelemetryServer:
    def __init__(
        self,
        host: str = config.TELEMETRY_HOST,
        port: int = config.TELEMETRY_PORT,
        auth: AuthService | None = None,
    ):
        self.host = host
        self.port = port
        self.auth = auth if auth is not None else AuthService()
        # Only authenticated connections are ever added here; broadcast()
        # therefore only reaches authenticated clients. Guarded by
        # _clients_lock because the asyncio thread mutates it while the
        # pipeline's main thread iterates it in broadcast().
        self.connected_clients = set()
        # Connections already dropped for being too slow to absorb a
        # broadcast/reply; guarded by the same _clients_lock. It only exists
        # so repeated send timeouts on one socket log/close it once.
        self._slow_clients = set()
        self._clients_lock = threading.Lock()
        # Shared failed-auth budget per client IP: (count, window_start).
        # Reserve-before-check makes the cap exact under parallel bursts;
        # refunded when the credential turns out to be correct.
        self._ip_failures: dict[str, tuple[int, float]] = {}
        self._ip_failures_lock = threading.Lock()
        # Successful-registration budget per client IP: timestamps (monotonic)
        # of the successes inside the rolling window, one entry per source IP.
        # Everything is per server instance, so every test that starts its own
        # TelemetryServer gets a fresh budget.
        self._ip_reg_successes: dict[str, list[float]] = {}
        self._ip_reg_successes_lock = threading.Lock()
        # Monotonic timestamps of the last full-map expiry sweep per budget;
        # guarded by the matching lock above (see IP_SWEEP_INTERVAL_SECONDS).
        self._last_failure_sweep = 0.0
        self._last_registration_sweep = 0.0
        # Opt-in TLS context, built in start() from LIVEGUARD_TLS_CERT/KEY.
        # None means the plain ws:// behaviour of earlier versions.
        self._ssl_context: ssl.SSLContext | None = None
        # Shutdown coordination for stop(): the future _serve awaits while
        # serving (created on the event loop, completed via
        # loop.call_soon_threadsafe), plus the flag covering stop() racing
        # start() before the loop even exists.
        self._serve_future: asyncio.Future | None = None
        self._stop_requested = threading.Event()
        # Startup readiness: set by _serve only AFTER websockets.serve has
        # successfully bound and entered (never when the bind fails), cleared
        # by start() before relaunching and by stop() once the server has
        # stopped. wait_ready() parks calling threads on this event, so a
        # caller can tell a serving server from one whose bind failure is
        # only visible in the log.
        self._ready = threading.Event()
        self.loop = None
        self.thread = None

    # ------------------------------------------------------------------
    # Message helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_message(message) -> tuple[dict | None, str]:
        """Return ``(payload, "")`` for a valid JSON object, else ``(None, reason)``."""
        if not isinstance(message, str):
            return None, "binary frames are not supported"
        try:
            data = json.loads(message)
        except (json.JSONDecodeError, ValueError):
            return None, "invalid JSON"
        if not isinstance(data, dict):
            return None, "expected a JSON object"
        if not isinstance(data.get("type"), str) or not data.get("type"):
            return None, "missing or invalid message type"
        return data, ""

    async def _send(self, websocket, payload: dict) -> None:
        """Send one direct reply, never blocking the event loop for long.

        Serializes the payload here, then defers the actual write to
        :meth:`_deliver`, which owns the single copy of the per-send timeout,
        slow-client drop and exception containment shared with broadcasts.
        Best-effort: on timeout the peer is dropped (it stopped reading), on
        disconnect the reply is simply lost.
        """
        await self._deliver(websocket, json.dumps(payload), context="reply")

    @staticmethod
    async def _close(websocket, reason: str, code: int = POLICY_VIOLATION_CLOSE_CODE) -> None:
        try:
            await websocket.close(code=code, reason=reason)
        except ConnectionClosed:
            pass  # already gone

    # ------------------------------------------------------------------
    # Failed-attempt budget (per client IP, survives reconnects)
    # ------------------------------------------------------------------
    @staticmethod
    def _client_ip(websocket) -> str:
        addr = getattr(websocket, "remote_address", None)
        return addr[0] if isinstance(addr, tuple) and addr else "unknown"

    def _reserve_ip_attempt(self, ip: str) -> bool:
        """Atomically reserve one failed-attempt slot for ``ip``.

        Returns False when this IP's budget is exhausted (pre-deny, close
        1008). Because the reservation happens *before* the password is
        evaluated and increments only while count < MAX, concurrent bursts
        cannot overrun the cap: the extra connections are refused without
        getting a guess. Successful logins call :meth:`_refund_ip_attempt`.
        Expired entries are swept opportunistically so a long-running server
        only ever holds one key per active source IP: the full-map sweep runs
        at most once per IP_SWEEP_INTERVAL_SECONDS, while the entry for
        ``ip`` itself is always expiry-filtered, so this IP's window stays
        exact between sweeps.
        """
        now = time.monotonic()
        with self._ip_failures_lock:
            if now - self._last_failure_sweep >= IP_SWEEP_INTERVAL_SECONDS:
                self._last_failure_sweep = now
                expired = [
                    key
                    for key, (_, started) in self._ip_failures.items()
                    if now - started > IP_FAILURE_WINDOW_SECONDS
                ]
                for key in expired:
                    del self._ip_failures[key]
            entry = self._ip_failures.get(ip)
            if entry is not None and now - entry[1] > IP_FAILURE_WINDOW_SECONDS:
                # The queried key's window is expiry-filtered on EVERY reserve
                # (not only during the throttled sweep): an expired entry
                # starts a fresh window exactly as if it had just been swept.
                del self._ip_failures[ip]
                entry = None
            count, started = entry if entry is not None else (0, now)
            if count >= MAX_IP_AUTH_FAILURES:
                return False
            self._ip_failures[ip] = (count + 1, started)
            return True

    def _refund_ip_attempt(self, ip: str) -> None:
        """Give back the slot reserved by a login that turned out valid."""
        with self._ip_failures_lock:
            entry = self._ip_failures.get(ip)
            if entry is None:
                return
            count, started = entry
            if count > 1:
                self._ip_failures[ip] = (count - 1, started)
            else:
                self._ip_failures.pop(ip, None)

    # ------------------------------------------------------------------
    # Successful-registration budget (per client IP, per server instance)
    # ------------------------------------------------------------------
    def _reserve_ip_registration(self, ip: str) -> bool:
        """Atomically reserve one *successful*-registration slot for ``ip``.

        Reserved before the account is created, refunded when creation fails,
        so the rolling window only ever counts successes and a parallel burst
        cannot overshoot the cap. Expired stamps are swept opportunistically:
        the full-map sweep runs at most once per IP_SWEEP_INTERVAL_SECONDS,
        while the stamps for ``ip`` itself are always expiry-filtered, so this
        IP's window stays exact between sweeps.
        """
        now = time.monotonic()
        with self._ip_reg_successes_lock:
            if now - self._last_registration_sweep >= IP_SWEEP_INTERVAL_SECONDS:
                self._last_registration_sweep = now
                for key in list(self._ip_reg_successes):
                    active = [
                        stamp
                        for stamp in self._ip_reg_successes[key]
                        if now - stamp <= IP_REGISTER_WINDOW_SECONDS
                    ]
                    if active:
                        self._ip_reg_successes[key] = active
                    else:
                        del self._ip_reg_successes[key]
            # Always expiry-filter the queried key (even when the throttled
            # sweep above didn't run), so a window that has just elapsed is
            # honoured exactly on the very next reserve for that IP.
            stamps = [
                stamp
                for stamp in self._ip_reg_successes.get(ip, [])
                if now - stamp <= IP_REGISTER_WINDOW_SECONDS
            ]
            if stamps:
                self._ip_reg_successes[ip] = stamps
            else:
                self._ip_reg_successes.pop(ip, None)
            if len(stamps) >= MAX_IP_REGISTER_SUCCESSES:
                return False
            stamps.append(now)
            self._ip_reg_successes[ip] = stamps
            return True

    def _refund_ip_registration(self, ip: str) -> None:
        """Give back the slot reserved by a registration that failed.

        Reservations are interchangeable (they only carry the count), so
        dropping the newest stamp is correct even under concurrency.
        """
        with self._ip_reg_successes_lock:
            stamps = self._ip_reg_successes.get(ip)
            if not stamps:
                return
            stamps.pop()
            if not stamps:
                self._ip_reg_successes.pop(ip, None)

    # ------------------------------------------------------------------
    # Slow-client protection
    # ------------------------------------------------------------------
    async def _drop_slow_client(self, websocket, context: str) -> None:
        """Disconnect a client that stopped absorbing data.

        Idempotent: the first timeout for a socket logs it and starts the
        closing handshake; later timeouts for the same socket (messages that
        were already queued) find it unregistered and do nothing. Once dropped
        the socket is out of ``connected_clients``, so no further broadcasts
        are ever buffered for it.
        """
        with self._clients_lock:
            self.connected_clients.discard(websocket)
            first = websocket not in self._slow_clients
            if first:
                self._slow_clients.add(websocket)
        if not first:
            return
        client_ip = self._client_ip(websocket)
        logger.warning(
            "slow client %s: %s send exceeded %.1fs; disconnecting",
            client_ip,
            context,
            CLIENT_SEND_TIMEOUT,
        )
        try:
            await asyncio.wait_for(
                self._close(websocket, "slow client"),
                timeout=SLOW_CLIENT_CLOSE_TIMEOUT,
            )
        except (TimeoutError, asyncio.TimeoutError):
            # The peer is not reading, so the closing handshake cannot
            # finish. Abort the TCP connection so the handler wakes up and
            # cleans the connection out immediately.
            logger.warning(
                "slow client %s: close handshake timed out; aborting TCP connection",
                client_ip,
            )
            transport = getattr(websocket, "transport", None)
            if transport is not None:
                transport.abort()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    async def _auth_phase(self, websocket, conn_state: dict | None = None) -> tuple[str, str] | None:
        """Wait (max 10s) for a successful auth handshake.

        Returns ``(username, token)`` on success, or ``None`` if the
        connection was closed (timeout, rate limit, or peer disconnect).
        Registration messages are served here too but never authenticate.

        ``conn_state`` is per connection (created by :meth:`_handler`), so
        registration-success caps survive logout/re-auth on the same socket.
        """
        if conn_state is None:
            conn_state = {"register_successes": 0}
        failures = 0
        register_failures = 0
        deadline = time.monotonic() + AUTH_TIMEOUT_SECONDS
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                await self._close(websocket, "authentication timeout")
                return None
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=remaining)
            # On Python 3.10 asyncio.TimeoutError is NOT the builtin
            # TimeoutError (aliases only from 3.11), so catch both or the
            # auth timeout escapes to the generic handler and the client is
            # dropped with an empty close reason.
            except (TimeoutError, asyncio.TimeoutError):
                await self._close(websocket, "authentication timeout")
                return None
            except ConnectionClosed:
                return None

            data, reason = self._parse_message(message)
            if data is None:
                await self._send(
                    websocket,
                    {"type": "error", "code": "bad_message", "reason": reason},
                )
                continue

            message_type = data["type"]

            if message_type == "register":
                # Abuse control: cap registration *successes* (failures are
                # limited per connection below) -- one per connection, plus a
                # generous per-IP rolling window so bulk account creation
                # from one address is slowed down. Both caps are checked
                # before the account is created, so a denied attempt never
                # writes anything.
                client_ip = self._client_ip(websocket)
                if (
                    conn_state.get("register_successes", 0)
                    >= MAX_REGISTER_SUCCESSES_PER_CONNECTION
                ):
                    logger.warning(
                        "registration denied for %s: per-connection success limit reached",
                        client_ip,
                    )
                    await self._send(
                        websocket,
                        {"type": "error", "code": "register_failed",
                         "reason": "registration limit reached"},
                    )
                    continue
                if not self._reserve_ip_registration(client_ip):
                    logger.warning(
                        "registration denied for %s: per-IP success limit reached",
                        client_ip,
                    )
                    await self._send(
                        websocket,
                        {"type": "error", "code": "register_failed",
                         "reason": "too many registrations"},
                    )
                    continue
                username = data.get("username")
                password = data.get("password")
                # PBKDF2 runs off the event loop (asyncio.to_thread) so a
                # register/login storm cannot stall live telemetry delivery.
                try:
                    ok, reason = await asyncio.to_thread(
                        self.auth.register, username, password
                    )
                except Exception:
                    # The IP registration slot was reserved above; refund it
                    # or a failing store (locked database, disk full, ...)
                    # would leak one slot per crash for the whole window.
                    # Re-raise: _handler's catch-all kills the connection.
                    self._refund_ip_registration(client_ip)
                    raise
                if ok:
                    conn_state["register_successes"] = (
                        conn_state.get("register_successes", 0) + 1
                    )
                    normalized = username.strip().lower() if isinstance(username, str) else ""
                    await self._send(
                        websocket, {"type": "register_ok", "username": normalized}
                    )
                else:
                    self._refund_ip_registration(client_ip)
                    register_failures += 1
                    await asyncio.sleep(AUTH_FAILURE_DELAY)
                    if register_failures >= MAX_REGISTER_FAILURES:
                        await self._send(
                            websocket,
                            {"type": "error", "code": "register_failed",
                             "reason": "too many attempts"},
                        )
                        await self._close(websocket, "too many attempts")
                        return None
                    await self._send(
                        websocket,
                        {"type": "error", "code": "register_failed", "reason": reason},
                    )
                continue

            if message_type == "auth":
                failure_reason = "invalid credentials"
                result: tuple[str, str] | None = None
                # Budget is per client IP and reserved before the password is
                # evaluated, so parallel connections cannot overrun the cap.
                client_ip = self._client_ip(websocket)
                if not self._reserve_ip_attempt(client_ip):
                    logger.warning(
                        "auth rate limit hit for %s: budget exhausted", client_ip
                    )
                    await self._send(
                        websocket,
                        {"type": "auth_error", "reason": "too many attempts"},
                    )
                    await self._close(websocket, "too many attempts")
                    return None
                if "token" in data:
                    token = data.get("token")
                    try:
                        username = (
                            self.auth.verify_token(token)
                            if isinstance(token, str) else None
                        )
                    except Exception:
                        # The failed-attempt slot was reserved before the
                        # token check; refund it so a broken store
                        # cannot silently leak rate-limit budget. Re-raise:
                        # _handler's catch-all kills the connection.
                        self._refund_ip_attempt(client_ip)
                        raise
                    if username is not None:
                        result = (username, token)
                    else:
                        failure_reason = "invalid or expired token"
                else:
                    username = data.get("username")
                    password = data.get("password")
                    token = None
                    if isinstance(username, str) and isinstance(password, str):
                        # PBKDF2 off the event loop -- see register branch.
                        try:
                            token = await asyncio.to_thread(
                                self.auth.authenticate, username, password
                            )
                        except Exception:
                            # The failed-attempt slot was reserved before the
                            # password check; refund it so a broken store
                            # cannot silently leak rate-limit budget. Re-raise:
                            # _handler's catch-all kills the connection.
                            self._refund_ip_attempt(client_ip)
                            raise
                    if token is not None:
                        result = (username.strip().lower(), token)

                if result is not None:
                    # Valid credential: refund the reservation (budget tracks
                    # failures only) and admit.
                    self._refund_ip_attempt(client_ip)
                    # auth_ok itself is sent by _handler, after this connection
                    # has been registered as authenticated.
                    return result

                failures += 1
                logger.warning(
                    "auth failed (%s) for %r from %s",
                    failure_reason,
                    username,
                    client_ip,
                )
                await asyncio.sleep(AUTH_FAILURE_DELAY)
                if failures >= MAX_AUTH_FAILURES:
                    await self._send(
                        websocket,
                        {"type": "auth_error", "reason": "too many attempts"},
                    )
                    await self._close(websocket, "too many attempts")
                    return None
                await self._send(
                    websocket, {"type": "auth_error", "reason": failure_reason}
                )
                continue

            await self._send(
                websocket,
                {"type": "error", "code": "bad_message", "reason": "unexpected message type"},
            )

    async def _note_bad_session_message(self, websocket, conn_state: dict) -> bool:
        """Count one more consecutive bad message in the session phase.

        Returns True once the per-connection budget is exhausted, after
        closing the connection with 1008. The counter lives in the
        per-connection state, so it survives logout/re-auth cycles on the
        same socket, and it is reset whenever a recognized message (logout)
        arrives -- without that reset, an honest client that occasionally
        misbehaves would eventually be locked out.
        """
        bad = conn_state.get("bad_messages", 0) + 1
        conn_state["bad_messages"] = bad
        if bad < MAX_SESSION_BAD_MESSAGES:
            return False
        logger.warning(
            "closing %s after %d consecutive bad session messages",
            self._client_ip(websocket),
            bad,
        )
        await self._close(websocket, "too many bad messages")
        return True

    async def _session_phase(
        self, websocket, conn_state: dict | None = None
    ) -> str:
        """Serve an authenticated client until logout or disconnect.

        Returns ``"logged_out"`` (client may re-authenticate) or ``"closed"``.

        Malformed and unknown frames are answered with one error each, but
        only up to MAX_SESSION_BAD_MESSAGES *consecutive* ones: a client that
        spams junk can no longer earn freshly serialized error replies
        forever. The counter is kept in ``conn_state`` (per connection) and
        reset by every recognized message.
        """
        if conn_state is None:
            conn_state = {}
        while True:
            try:
                message = await websocket.recv()
            except ConnectionClosed:
                return "closed"

            data, reason = self._parse_message(message)
            if data is None:
                if await self._note_bad_session_message(websocket, conn_state):
                    return "closed"
                await self._send(
                    websocket,
                    {"type": "error", "code": "bad_message", "reason": reason},
                )
                continue

            if data["type"] == "logout":
                # Stop broadcasts before confirming; a client that has read
                # "logged_out" is guaranteed to be out of connected_clients.
                # logout is recognized -> reset the bad-message budget.
                conn_state["bad_messages"] = 0
                with self._clients_lock:
                    self.connected_clients.discard(websocket)
                await self._send(websocket, {"type": "logged_out"})
                return "logged_out"

            if await self._note_bad_session_message(websocket, conn_state):
                return "closed"
            await self._send(
                websocket,
                {"type": "error", "code": "bad_message", "reason": "unexpected message type"},
            )

    async def _handler(self, websocket):
        # Per-connection state: deliberately created once per socket, so the
        # registration-success cap holds across logout/re-auth cycles too.
        conn_state: dict = {"register_successes": 0}
        try:
            while True:
                result = await self._auth_phase(websocket, conn_state)
                if result is None:
                    return
                username, token = result
                # Register as authenticated, then queue auth_ok. The client
                # ignores stream frames until it has read auth_ok, so any
                # broadcast that races ahead is discarded client-side.
                with self._clients_lock:
                    self.connected_clients.add(websocket)
                await self._send(
                    websocket,
                    {"type": "auth_ok", "token": token, "username": username},
                )
                outcome = await self._session_phase(websocket, conn_state)
                if outcome != "logged_out":
                    return
                # logged out -> back to the authentication phase
        except ConnectionClosed:
            pass  # normal disconnect
        except Exception:
            # A misbehaving client must never take the server down.
            logger.exception("telemetry websocket handler failed")
        finally:
            with self._clients_lock:
                self.connected_clients.discard(websocket)
                self._slow_clients.discard(websocket)

    # ------------------------------------------------------------------
    # Server plumbing (unchanged public behaviour)
    # ------------------------------------------------------------------
    @staticmethod
    def _build_ssl_context() -> ssl.SSLContext | None:
        """Build the opt-in TLS context from LIVEGUARD_TLS_CERT/KEY.

        Returns ``None`` when both are unset (plain ws://, exactly as before).
        Raises when only one is set or the PEM files cannot be loaded, so a
        requested TLS server fails closed instead of silently downgrading.
        """
        cert = os.environ.get("LIVEGUARD_TLS_CERT")
        key = os.environ.get("LIVEGUARD_TLS_KEY")
        if not cert and not key:
            return None
        if not cert or not key:
            raise ValueError(
                "LIVEGUARD_TLS_CERT and LIVEGUARD_TLS_KEY must both be set"
            )
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=cert, keyfile=key)
        return context

    async def _serve(self):
        self.loop = asyncio.get_running_loop()
        # Created here (on this loop) so stop() can complete it thread-safely
        # from any other thread; cleared again on the way out.
        self._serve_future = self.loop.create_future()
        if self._stop_requested.is_set() and not self._serve_future.done():
            # stop() won the race with start(): never begin serving.
            self._serve_future.set_result(None)
        try:
            async with websockets.serve(
                self._handler, self.host, self.port, ssl=self._ssl_context,
                # --- pinned websockets defaults -------------------------------
                # Verified against the installed websockets 17.1 (the
                # dependency is `websockets>=12.0`, unbounded) and every one
                # of these kwargs exists with the same meaning at that floor.
                # Passing them explicitly means a future release that changes a
                # default surfaces here as a reviewed diff instead of silent
                # runtime drift. Version-dependent defaults (server_header)
                # and non-behavioural ones (logger) are deliberately not
                # pinned.
                ping_interval=20,       # keepalive ping every 20s (None = off)
                ping_timeout=20,        # drop the peer if no pong within 20s
                close_timeout=10,       # wait up to 10s for closing handshake
                open_timeout=10,        # HTTP upgrade handshake timeout (s)
                max_size=1048576,       # 1 MiB max incoming frame
                max_queue=16,           # frames buffered per connection
                write_limit=32768,      # 32 KiB per-connection write buffer
                compression="deflate",  # permessage-deflate negotiated
            ):
                if self._ssl_context is not None:
                    # Status banner, TLS mode only: plain ws:// output stays
                    # byte-identical to earlier versions.
                    print(
                        f"[Status] telemetry server listening on wss://{self.host}:{self.port}",
                        flush=True,
                    )
                # The listener is bound and accepting: only NOW is the server
                # actually serving, so only now may wait_ready() be told so.
                # A bind failure escapes before this line, so _ready stays
                # clear and wait_ready() reports False.
                self._ready.set()
                await self._serve_future
        finally:
            self._serve_future = None

    def start(self):
        self._stop_requested.clear()  # a fresh start() supersedes any stop()
        self._ready.clear()  # restart hygiene: ready only once THIS start binds
        try:
            self._ssl_context = self._build_ssl_context()
        except Exception as exc:
            # Fail closed: never serve plaintext when TLS was requested.
            logger.error("TLS setup failed; telemetry server not started: %s", exc)
            return

        def run_loop():
            try:
                asyncio.run(self._serve())
            except Exception as exc:
                logger.error("telemetry server failed: %s", exc, exc_info=True)

        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()

    def wait_ready(self, timeout: float | None = None) -> bool:
        """Block until this server is actually serving, or ``timeout`` elapses.

        Thread-safe (call it from any thread): parks the caller on the
        internal readiness event that :meth:`_serve` sets only *after*
        ``websockets.serve`` has successfully bound and entered, so it
        returns True exactly when a client can connect. Returns False on
        timeout, before ``start()``, when the bind failed (the error is
        already logged by the serving thread), and after ``stop()``.
        ``timeout=None`` waits forever -- only sensible when something else
        guarantees the server will come up.
        """
        return self._ready.wait(timeout)

    def stop(self, timeout: float = 3.0) -> None:
        """Stop a server started by :meth:`start()` and join its thread.

        Thread-safe (call it from any thread), idempotent, and safe to call
        when the server was never started. Completing the serving future
        exits ``_serve``'s ``async with`` block, which closes the listener,
        closes every open connection (close code 1001) and waits for the
        connection handlers -- so the thread only ends once everything is
        cleaned up. Returns when the thread has ended or ``timeout`` seconds
        have elapsed; a still-shutting-down thread is left to finish (it is
        a daemon thread, so it can never block process exit). Clears the
        readiness event, so :meth:`wait_ready` reports False after stop().
        """
        self._stop_requested.set()  # covers stop() racing a slow start()
        future = self._serve_future
        loop = self.loop
        if future is not None and not future.done() and loop is not None:
            try:
                # Futures may only be completed from their own event loop;
                # hand the wake-up to the loop itself.
                loop.call_soon_threadsafe(self._finish_serving, future)
            except RuntimeError:
                pass  # loop already closed; the thread is on its way down
        thread = self.thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
        # The server has stopped (or never got far enough to serve): readiness
        # must not outlive stop(), so a later wait_ready() reports False and a
        # subsequent start() begins from a known-clean state (start() clears
        # it too, before relaunching).
        self._ready.clear()

    @staticmethod
    def _finish_serving(future: asyncio.Future) -> None:
        """Complete ``future``; always runs on the owning event-loop thread."""
        if not future.done():
            future.set_result(None)

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------
    async def _deliver(
        self, websocket, message: str, context: str = "broadcast"
    ) -> None:
        """Send one already-serialized message to one client.

        The single delivery path shared by direct replies (:meth:`_send`,
        ``context="reply"``) and broadcast fan-out (:meth:`_fanout`).
        Bounded by CLIENT_SEND_TIMEOUT and exception-safe: delivery is
        best-effort per client, so a dead, slow or misbehaving peer only
        ever affects itself.
        """
        try:
            await asyncio.wait_for(
                websocket.send(message), timeout=CLIENT_SEND_TIMEOUT
            )
        # asyncio.TimeoutError is only an alias of the builtin TimeoutError
        # from Python 3.11; on 3.10 it is a distinct class, so catch both.
        except (TimeoutError, asyncio.TimeoutError):
            await self._drop_slow_client(websocket, context)
        except ConnectionClosed:
            pass  # client went away; the handler cleans it up
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("%s delivery failed (%s): %r",
                           context, self._client_ip(websocket), exc)

    async def _fanout(self, message: str, targets: list) -> None:
        """Hand the single serialized message to every target concurrently.

        Each connection gets its own task, so one client blocked in its TCP
        write can never delay delivery to the others; gather() only exists to
        keep the tasks referenced and to contain every exception.
        """
        tasks = [asyncio.create_task(self._deliver(ws, message)) for ws in targets]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def broadcast(self, payload: dict):
        """Best-effort fan-out to every authenticated client.

        The payload is serialized exactly once and the *same* string is handed
        to every connection. The whole fan-out is a single coroutine scheduled
        on the event loop (one wakeup per broadcast instead of one per
        client), which then fans out per-connection tasks with per-send
        timeouts, so neither the caller nor healthy clients ever wait on a
        slow one.
        """
        loop = self.loop
        if loop is None or loop.is_closed():
            return
        with self._clients_lock:
            targets = list(self.connected_clients)
        if not targets:
            return

        message = json.dumps(payload)
        try:
            asyncio.run_coroutine_threadsafe(self._fanout(message, targets), loop)
        except RuntimeError:
            pass  # loop is shutting down; nothing to deliver to
