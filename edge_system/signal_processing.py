import collections
import numpy as np
from scipy import signal
from edge_system import config


class RealTimeFilter:
    def __init__(self, fs: int = config.SAMPLING_RATE_ECG):
        self.fs = fs
        low = config.FILTER_LOWCUT / (0.5 * fs)
        high = config.FILTER_HIGHCUT / (0.5 * fs)
        self.b_band, self.a_band = signal.butter(
            config.FILTER_ORDER, [low, high], btype="bandpass"
        )
        self.zi_band = signal.lfilter_zi(self.b_band, self.a_band) * 0.0

        w0 = config.NOTCH_FREQ / (0.5 * fs)
        self.b_notch, self.a_notch = signal.iirnotch(w0, config.NOTCH_Q)
        self.zi_notch = signal.lfilter_zi(self.b_notch, self.a_notch) * 0.0

    def process_sample(self, raw_sample: float) -> float:
        filtered_band, self.zi_band = signal.lfilter(
            self.b_band, self.a_band, [raw_sample], zi=self.zi_band
        )
        filtered_notch, self.zi_notch = signal.lfilter(
            self.b_notch, self.a_notch, filtered_band, zi=self.zi_notch
        )
        return float(filtered_notch[0])


class PanTompkinsQRSDetector:
    def __init__(self, fs: int = config.SAMPLING_RATE_ECG):
        self.fs = fs
        self.refractory_period = config.QRS_REFRACTORY_PERIOD
        self.samples_since_last_peak = self.refractory_period

        self.int_window = config.INTEGRATION_WINDOW
        self.int_buffer = collections.deque(maxlen=self.int_window)
        self.deriv_buffer = collections.deque(maxlen=5)

        self.signal_level = 0.0
        self.noise_level = 0.0
        self.threshold_i1 = 0.0
        self.threshold_i2 = 0.0
        self.recent_rr = collections.deque(maxlen=8)

    def process_sample(self, filtered_sample: float) -> bool:
        self.samples_since_last_peak += 1
        self.deriv_buffer.append(filtered_sample)

        if len(self.deriv_buffer) < 5:
            return False

        d = (
            2.0 * self.deriv_buffer[4]
            + self.deriv_buffer[3]
            - self.deriv_buffer[1]
            - 2.0 * self.deriv_buffer[0]
        ) / 8.0

        squared = d * d
        self.int_buffer.append(squared)
        integrated = sum(self.int_buffer) / len(self.int_buffer)

        is_peak = False
        if self.samples_since_last_peak > self.refractory_period:
            if integrated > self.threshold_i1:
                is_peak = True
                self.signal_level = 0.125 * integrated + 0.875 * self.signal_level
                self.recent_rr.append(self.samples_since_last_peak)
                self.samples_since_last_peak = 0
            else:
                self.noise_level = 0.125 * integrated + 0.875 * self.noise_level

            self.threshold_i1 = self.noise_level + 0.25 * (
                self.signal_level - self.noise_level
            )
            self.threshold_i2 = 0.5 * self.threshold_i1

        return is_peak

    def get_heart_rate(self) -> float:
        if len(self.recent_rr) < 2:
            return 72.0
        avg_rr_samples = np.mean(self.recent_rr)
        bpm = (self.fs * 60.0) / max(1.0, avg_rr_samples)
        return float(np.clip(bpm, 40.0, 220.0))


class BeatSegmenter:
    def __init__(
        self,
        pre_r: int = config.PRE_R_SAMPLES,
        post_r: int = config.POST_R_SAMPLES,
        window_size: int = config.BEAT_WINDOW_SIZE,
    ):
        self.pre_r = pre_r
        self.post_r = post_r
        self.window_size = window_size
        self.raw_buffer = collections.deque(maxlen=window_size + 100)
        self.samples_since_peak = None

    def add_sample(self, filtered_sample: float, is_r_peak: bool):
        self.raw_buffer.append(filtered_sample)

        if is_r_peak:
            self.samples_since_peak = 0

        if self.samples_since_peak is not None:
            self.samples_since_peak += 1

            if self.samples_since_peak == self.post_r:
                self.samples_since_peak = None
                total_needed = self.pre_r + self.post_r
                if len(self.raw_buffer) >= total_needed:
                    beat = np.array(
                        list(self.raw_buffer)[-total_needed:], dtype=np.float32
                    )
                    std = np.std(beat)
                    if std > 1e-6:
                        beat = (beat - np.mean(beat)) / std
                    else:
                        beat = beat - np.mean(beat)

                    return beat.reshape(1, self.window_size)

        return None
