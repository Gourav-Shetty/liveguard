import sys
import time
import serial
import serial.tools.list_ports
from backend import config


class ArduinoSerialDriver:
    def __init__(self, port: str = None, baudrate: int = config.SERIAL_BAUDRATE):
        self.baudrate = baudrate
        self.port = port or self.auto_detect_port()
        self.ser = None
        self._last_read_error_log = None  # time.monotonic() of last logged read error

    def auto_detect_port(self) -> str:
        ports = list(serial.tools.list_ports.comports())
        for p in ports:
            desc = p.description.lower()
            if "arduino" in desc or "ch340" in desc or "usb-serial" in desc or "cp210" in desc:
                return p.device

        return config.SERIAL_PORT_WINDOWS if sys.platform.startswith("win") else config.SERIAL_PORT_LINUX

    def connect(self):
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=config.SERIAL_TIMEOUT
            )
            time.sleep(1.5)
            self.ser.reset_input_buffer()
        except Exception as e:
            self.ser = None
            raise e

    def read_sample(self):
        if self.ser is None or not self.ser.is_open:
            return None

        try:
            line = self.ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                return None

            parts = line.split(",")
            if len(parts) >= 2:
                ecg_raw = int(parts[0])
                leads_off = bool(int(parts[1]))
                ppg_ir = int(parts[2]) if len(parts) > 2 else 0
                ppg_red = int(parts[3]) if len(parts) > 3 else 0

                return {
                    "ecg_raw": ecg_raw,
                    "leads_off": leads_off,
                    "ppg_ir": ppg_ir,
                    "ppg_red": ppg_red,
                    "timestamp": time.time()
                }
        except Exception as exc:
            now = time.monotonic()
            if self._last_read_error_log is None or now - self._last_read_error_log >= 5.0:
                self._last_read_error_log = now
                print(f"[WARN] serial read error (suppressed): {exc!r}", file=sys.stderr, flush=True)
            return None

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
