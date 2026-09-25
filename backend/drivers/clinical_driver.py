import os
import time
import csv
import numpy as np
from backend import config

class ClinicalPatientDriver:
    def __init__(self, patient_id: str = "100"):
        self.patient_id = str(patient_id)
        self.csv_path = self._find_csv()
        self.interval = 1.0 / config.SAMPLING_RATE_ECG
        self.index = 0
        self.signal = None
        self.start_time = None
        self.sample_idx = 0
        self._load_signal()

    def _find_csv(self):
        possible_dirs = []
        env_dir = os.environ.get("LIVEGUARD_ARCHIVE_DIR")
        if env_dir:
            possible_dirs.append(env_dir)
        possible_dirs.append(str(config.BASE_DIR / "archive" / "mitbih_database"))

        searched = []
        for d in possible_dirs:
            p = os.path.join(d, f"{self.patient_id}.csv")
            searched.append(p)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(
            f"Could not find {self.patient_id}.csv. Searched: {', '.join(searched)}"
        )

    def _load_signal(self):
        values = []
        with open(self.csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            _ = next(reader, None)
            for row in reader:
                if len(row) > 1:
                    try:
                        values.append(float(row[1]))
                    except ValueError:
                        continue
        self.signal = np.array(values, dtype=np.float32)
        print(f"[Clinical Driver] Loaded Patient {self.patient_id} ({len(self.signal)} samples @ 360Hz)")

    def read_sample(self):
        now = time.time()
        if self.start_time is None:
            self.start_time = now
            self.sample_idx = 0

        # High-precision pacing every 10 samples to avoid OS timer sleep overhead
        if self.sample_idx % 10 == 0:
            target_time = self.start_time + self.sample_idx * self.interval
            sleep_needed = target_time - now
            if sleep_needed > 0:
                time.sleep(sleep_needed)

        self.sample_idx += 1
        if self.index >= len(self.signal):
            self.index = 0
            self.start_time = time.time()
            self.sample_idx = 0

        raw_val = self.signal[self.index]
        self.index += 1

        return {
            "ecg_raw": float(raw_val),
            "leads_off": False,
            "ppg_ir": 82000,
            "ppg_red": 77000,
            "timestamp": time.time()
        }

    def close(self):
        pass
