import asyncio
import json
import threading
import websockets
from edge_system import config


class TelemetryServer:
    def __init__(self, host: str = config.TELEMETRY_HOST, port: int = config.TELEMETRY_PORT):
        self.host = host
        self.port = port
        self.connected_clients = set()
        self.loop = None
        self.thread = None

    async def _register(self, websocket):
        self.connected_clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.connected_clients.discard(websocket)

    async def _serve(self):
        self.loop = asyncio.get_running_loop()
        async with websockets.serve(self._register, self.host, self.port):
            await asyncio.Future()

    def start(self):
        def run_loop():
            try:
                asyncio.run(self._serve())
            except Exception:
                pass

        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()

    def broadcast(self, payload: dict):
        if not self.connected_clients or self.loop is None:
            return

        message = json.dumps(payload)
        for ws in list(self.connected_clients):
            try:
                asyncio.run_coroutine_threadsafe(ws.send(message), self.loop)
            except Exception:
                pass
