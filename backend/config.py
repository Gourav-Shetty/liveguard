import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("LIVEGUARD_DATA_DIR", BASE_DIR / "data"))

DATA_SOURCE = os.environ.get("LIVEGUARD_SOURCE", "MOCK")

SAMPLING_RATE_ECG = 360
SAMPLING_RATE_PPG = 50

BEAT_WINDOW_SIZE = 180
PRE_R_SAMPLES = 70
POST_R_SAMPLES = 110

FILTER_LOWCUT = 0.5
FILTER_HIGHCUT = 40.0
FILTER_ORDER = 4
NOTCH_FREQ = 50.0
NOTCH_Q = 30.0

QRS_REFRACTORY_PERIOD = 72
INTEGRATION_WINDOW = 54

SERIAL_PORT_WINDOWS = "COM3"
SERIAL_PORT_LINUX = "/dev/ttyACM0"
SERIAL_BAUDRATE = 115200
SERIAL_TIMEOUT = 1.0

SPI_BUS = 0
SPI_DEVICE = 0
MCP3008_ECG_CHANNEL = 0
GPIO_LEADS_OFF_PLUS = 23
GPIO_LEADS_OFF_MINUS = 24
I2C_BUS = 1
MAX30102_I2C_ADDR = 0x57

def _load_calibrated_threshold(fallback=0.35) -> float:
    """Return the calibrated threshold from <DATA_DIR>/stage1_threshold.json,
    or `fallback` if the file is missing or unreadable (never raise)."""
    path = DATA_DIR / "stage1_threshold.json"
    if not path.is_file():
        return fallback
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return float(data["threshold"])
    except Exception as exc:
        print(
            f"[WARN] could not load calibrated threshold from {path}: {exc}; "
            f"using fallback {fallback}",
            file=sys.stderr,
        )
        return fallback


DEFAULT_ANOMALY_THRESHOLD = _load_calibrated_threshold()

TELEMETRY_HOST = "0.0.0.0"
TELEMETRY_PORT = 8765
STREAM_BATCH_SIZE = 10
