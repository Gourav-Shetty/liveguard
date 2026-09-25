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
        # Opt-in TLS context, built in start() from LIVEGUARD_TLS_CERT/KEY.
        # None means the plain ws:// behaviour of earlier versions.
        self._ssl_context: ssl.SSLContext | None = None
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

        Best-effort: on timeout the peer is dropped (it stopped reading), on
        disconnect the reply is simply lost.
        """
        try:
            await asyncio.wait_for(
                websocket.send(json.dumps(payload)), timeout=CLIENT_SEND_TIMEOUT
            )
        except TimeoutError:
            await self._drop_slow_client(websocket, "reply")
        except ConnectionClosed:
            pass  # client went away before reading the reply

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
        only ever holds one key per active source IP.
        """
        now = time.monotonic()
        with self._ip_failures_lock:
            expired = [
                key
                for key, (_, started) in self._ip_failures.items()
                if now - started > IP_FAILURE_WINDOW_SECONDS
            ]
            for key in expired:
                del self._ip_failures[key]
            count, started = self._ip_failures.get(ip, (0, now))
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
        cannot overshoot the cap. Expired stamps are swept opportunistically.
        """
        now = time.monotonic()
        with self._ip_reg_successes_lock:
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
            stamps = self._ip_reg_successes.get(ip, [])
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
        except TimeoutError:
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
            except TimeoutError:
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
                ok, reason = await asyncio.to_thread(
                    self.auth.register, username, password
                )
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
                    username = (
                        self.auth.verify_token(token) if isinstance(token, str) else None
                    )
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
                        token = await asyncio.to_thread(
                            self.auth.authenticate, username, password
                        )
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

    async def _session_phase(self, websocket) -> str:
        """Serve an authenticated client until logout or disconnect.

        Returns ``"logged_out"`` (client may re-authenticate) or ``"closed"``.
        """
        while True:
            try:
                message = await websocket.recv()
            except ConnectionClosed:
                return "closed"

            data, reason = self._parse_message(message)
            if data is None:
                await self._send(
                    websocket,
                    {"type": "error", "code": "bad_message", "reason": reason},
                )
                continue

            if data["type"] == "logout":
                # Stop broadcasts before confirming; a client that has read
                # "logged_out" is guaranteed to be out of connected_clients.
                with self._clients_lock:
                    self.connected_clients.discard(websocket)
                await self._send(websocket, {"type": "logged_out"})
                return "logged_out"

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
                outcome = await self._session_phase(websocket)
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
        async with websockets.serve(
            self._handler, self.host, self.port, ssl=self._ssl_context
        ):
            if self._ssl_context is not None:
                # Status banner, TLS mode only: plain ws:// output stays
                # byte-identical to earlier versions.
                print(
                    f"[Status] telemetry server listening on wss://{self.host}:{self.port}",
                    flush=True,
                )
            await asyncio.Future()

    def start(self):
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

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------
    async def _deliver(self, websocket, message: str) -> None:
        """Send one already-serialized message to one client.

        Bounded by CLIENT_SEND_TIMEOUT and exception-safe: broadcast results
        are best-effort per client, so a dead or slow peer only ever affects
        itself.
        """
        try:
            await asyncio.wait_for(
                websocket.send(message), timeout=CLIENT_SEND_TIMEOUT
            )
        except TimeoutError:
            await self._drop_slow_client(websocket, "broadcast")
        except ConnectionClosed:
            pass  # client went away; the handler cleans it up
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("broadcast delivery failed (%s): %r",
                           self._client_ip(websocket), exc)

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
