import os
import time
import math
import numpy as np
from edge_system import config


class MockSensorDriver:
    def __init__(self, use_dataset: bool = True):
        self.use_dataset = use_dataset
        self.dataset_beats = None
        self.sample_idx = 0
        self.t = 0.0
        self.interval = 1.0 / config.SAMPLING_RATE_ECG

        if self.use_dataset:
            self._load_dataset()

    def _load_dataset(self):
        test_file = config.DATA_DIR / "mitbih_test_ready.npz"
        alt_file = config.BASE_DIR / "quickstart-pytorch" / "mitbih_test_ready.npz"

        for p in [test_file, alt_file]:
            if os.path.exists(p):
                try:
                    data = np.load(p, allow_pickle=True)
                    self.dataset_beats = data["X_cnn"]
                    self.dataset_labels = data["y"]
                    return
                except Exception:
                    pass

        self.dataset_beats = None

    def read_sample(self):
        time.sleep(self.interval)
        self.t += self.interval

        if self.dataset_beats is not None:
            beat_idx = (self.sample_idx // 180) % len(self.dataset_beats)
            in_beat_idx = self.sample_idx % 180
            val = float(self.dataset_beats[beat_idx, in_beat_idx, 0])
            self.sample_idx += 1
            ecg_raw = int(np.clip(512 + val * 120, 0, 1023))
        else:
            hr_hz = 1.2
            phase = (self.t * hr_hz) % 1.0

            baseline = 25.0 * math.sin(2.0 * math.pi * 0.25 * self.t)
            hum = 15.0 * math.sin(2.0 * math.pi * 50.0 * self.t)

            if 0.10 <= phase < 0.18:
                p_wave = 40.0 * math.sin(math.pi * (phase - 0.10) / 0.08)
            else:
                p_wave = 0.0

            if 0.28 <= phase < 0.32:
                q_wave = -35.0 * math.sin(math.pi * (phase - 0.28) / 0.04)
            else:
                q_wave = 0.0

            if 0.32 <= phase < 0.38:
                r_wave = 320.0 * math.sin(math.pi * (phase - 0.32) / 0.06)
            else:
                r_wave = 0.0

            if 0.38 <= phase < 0.42:
                s_wave = -50.0 * math.sin(math.pi * (phase - 0.38) / 0.04)
            else:
                s_wave = 0.0

            if 0.50 <= phase < 0.65:
                t_wave = 65.0 * math.sin(math.pi * (phase - 0.50) / 0.15)
            else:
                t_wave = 0.0

            ecg_signal = 512.0 + p_wave + q_wave + r_wave + s_wave + t_wave + baseline + hum
            ecg_raw = int(np.clip(ecg_signal, 0, 1023))

        return {
            "ecg_raw": ecg_raw,
            "leads_off": False,
            "ppg_ir": 80000,
            "ppg_red": 75000,
            "timestamp": time.time()
        }

    def close(self):
        pass
