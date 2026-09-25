import time
from backend import config

try:
    import smbus2
    import RPi.GPIO as GPIO
    RPI_AVAILABLE = True
except ImportError:
    RPI_AVAILABLE = False


class ADS1115Driver:
    def __init__(self, bus: int = 1, address: int = 0x48, channel: int = 0):
        if not RPI_AVAILABLE:
            raise RuntimeError("smbus2 or RPi.GPIO not available.")

        self.bus_num = bus
        self.address = address
        self.channel = channel
        self.bus = smbus2.SMBus(self.bus_num)

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(config.GPIO_LEADS_OFF_PLUS, GPIO.IN)
        GPIO.setup(config.GPIO_LEADS_OFF_MINUS, GPIO.IN)

    def read_sample(self):
        mux = (4 + self.channel) << 12
        pga = 1 << 9
        mode = 1 << 8
        data_rate = 7 << 5
        config_val = 0x8000 | mux | pga | mode | data_rate | 0x0003

        config_bytes = [(config_val >> 8) & 0xFF, config_val & 0xFF]
        self.bus.write_i2c_block_data(self.address, 0x01, config_bytes)

        time.sleep(0.0012)

        data = self.bus.read_i2c_block_data(self.address, 0x00, 2)
        raw_adc = (data[0] << 8) | data[1]
        if raw_adc > 32767:
            raw_adc -= 65536

        scaled_10bit = int((raw_adc + 32768) >> 6)
        ecg_raw = max(0, min(1023, scaled_10bit))

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
        if hasattr(self, "bus"):
            self.bus.close()
        if RPI_AVAILABLE:
            GPIO.cleanup()
