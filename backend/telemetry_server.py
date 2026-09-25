import asyncio
import json
import threading
import time

import websockets
from websockets.exceptions import ConnectionClosed

from backend import config
from backend.auth import AuthService

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
        self._clients_lock = threading.Lock()
        # Shared failed-auth budget per client IP: (count, window_start).
        # Reserve-before-check makes the cap exact under parallel bursts;
        # refunded when the credential turns out to be correct.
        self._ip_failures: dict[str, tuple[int, float]] = {}
        self._ip_failures_lock = threading.Lock()
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

    @staticmethod
    async def _send(websocket, payload: dict) -> None:
        try:
            await websocket.send(json.dumps(payload))
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
    # Connection lifecycle
    # ------------------------------------------------------------------
    async def _auth_phase(self, websocket) -> tuple[str, str] | None:
        """Wait (max 10s) for a successful auth handshake.

        Returns ``(username, token)`` on success, or ``None`` if the
        connection was closed (timeout, rate limit, or peer disconnect).
        Registration messages are served here too but never authenticate.
        """
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
                username = data.get("username")
                password = data.get("password")
                # PBKDF2 runs off the event loop (asyncio.to_thread) so a
                # register/login storm cannot stall live telemetry delivery.
                ok, reason = await asyncio.to_thread(
                    self.auth.register, username, password
                )
                if ok:
                    normalized = username.strip().lower() if isinstance(username, str) else ""
                    await self._send(
                        websocket, {"type": "register_ok", "username": normalized}
                    )
                else:
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
        try:
            while True:
                result = await self._auth_phase(websocket)
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
        except Exception as exc:
            # A misbehaving client must never take the server down.
            print(f"[ERROR] telemetry websocket handler failed: {exc!r}", flush=True)
        finally:
            with self._clients_lock:
                self.connected_clients.discard(websocket)

    # ------------------------------------------------------------------
    # Server plumbing (unchanged public behaviour)
    # ------------------------------------------------------------------
    async def _serve(self):
        self.loop = asyncio.get_running_loop()
        async with websockets.serve(self._handler, self.host, self.port):
            await asyncio.Future()

    def start(self):
        def run_loop():
            try:
                asyncio.run(self._serve())
            except Exception as exc:
                print(f"[ERROR] telemetry server failed: {exc}", flush=True)

        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()

    def broadcast(self, payload: dict):
        # connected_clients only ever contains authenticated connections.
        if self.loop is None:
            return
        with self._clients_lock:
            targets = list(self.connected_clients)
        if not targets:
            return

        message = json.dumps(payload)
        for ws in targets:
            try:
                asyncio.run_coroutine_threadsafe(ws.send(message), self.loop)
            except Exception:
                pass  # connection already gone; the handler cleans it up
