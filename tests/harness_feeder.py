"""Local feeder for tests/frontend_harness.js.

Starts a TelemetryServer and pushes synthetic telemetry/beat broadcasts so the
headless frontend harness can complete its live-stream checks without edge
hardware or a real patient feed.

Usage (from the repository root):
    python tests/harness_feeder.py 8766
then, in another shell:
    node tests/frontend_harness.js ws://127.0.0.1:8766

Set LIVEGUARD_DATA_DIR to isolate the auth store; wipe it before each run
(registration only opens while the user store is empty).
"""
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.telemetry_server import TelemetryServer


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8766
    server = TelemetryServer(host="127.0.0.1", port=port)
    server.start()

    def feeder():
        t0 = time.time()
        tick = 0
        while True:
            elapsed = time.time() - t0
            tick += 1
            server.broadcast({
                "type": "telemetry",
                "raw_ecg": 512,
                "filtered_ecg": round(100.0 * (1 if int(elapsed * 20) % 2 else -1), 2),
                "is_r_peak": False,
                "heart_rate": 60 + (tick % 30),
                "total_alerts": 0,
                "total_beats": 1,
                "timestamp": time.time(),
            })
            if int(elapsed * 2) % 2 == 0:
                server.broadcast({
                    "type": "beat",
                    "prediction": "Normal",
                    "is_anomaly": False,
                    "confidence": 98.5,
                    "abnormal_prob": 0.01,
                    "heart_rate": 60 + (tick % 30),
                    "total_alerts": 0,
                    "total_beats": 1,
                    "timestamp": time.time(),
                })
            time.sleep(0.05)

    threading.Thread(target=feeder, daemon=True).start()
    print(f"harness feeder listening on ws://127.0.0.1:{port}", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
