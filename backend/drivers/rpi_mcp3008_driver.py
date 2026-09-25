import time
from backend import config

try:
    import spidev
    import RPi.GPIO as GPIO
    RPI_AVAILABLE = True
except ImportError:
    RPI_AVAILABLE = False


class RPiMCP3008Driver:
    def __init__(
        self,
        bus: int = config.SPI_BUS,
        device: int = config.SPI_DEVICE,
        channel: int = config.MCP3008_ECG_CHANNEL,
    ):
        if not RPI_AVAILABLE:
            raise RuntimeError("spidev or RPi.GPIO not available.")

        self.bus = bus
        self.device = device
        self.channel = channel

        self.spi = spidev.SpiDev()
        self.spi.open(self.bus, self.device)
        self.spi.max_speed_hz = 1350000
        self.spi.mode = 0

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(config.GPIO_LEADS_OFF_PLUS, GPIO.IN)
        GPIO.setup(config.GPIO_LEADS_OFF_MINUS, GPIO.IN)

    def read_sample(self):
        adc_cmd = [1, (8 + self.channel) << 4, 0]
        reply = self.spi.xfer2(adc_cmd)
        ecg_raw = ((reply[1] & 3) << 8) + reply[2]

        lo_plus = GPIO.input(config.GPIO_LEADS_OFF_PLUS)
        lo_minus = GPIO.input(config.GPIO_LEADS_OFF_MINUS)
        leads_off = bool(lo_plus or lo_minus)

        return {
            "ecg_raw": ecg_raw,
            "leads_off": leads_off,
            "ppg_ir": 0,
            "ppg_red": 0,
            "timestamp": time.time(),
        }

    def close(self):
        if hasattr(self, "spi"):
            self.spi.close()
        if RPI_AVAILABLE:
            GPIO.cleanup()
