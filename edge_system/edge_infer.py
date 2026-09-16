import os
import numpy as np
from edge_system import config

class NumPyInferenceEngine:
    def __init__(self, weights_path: str = None, threshold: float = config.DEFAULT_ANOMALY_THRESHOLD):
        self.threshold = threshold
        self.weights = None
        base_dir = config.BASE_DIR
        npz_path = weights_path or str(base_dir / "quickstart-pytorch" / "stage1_weights.npz")
        self._load_weights(npz_path)

    def _load_weights(self, path: str):
        if os.path.exists(path):
            try:
                self.weights = dict(np.load(path))
                return
            except Exception:
                pass
        self._init_mock_weights()

    def _init_mock_weights(self):
        np.random.seed(42)
        self.weights = {
            "c1_w": np.random.randn(16, 1, 7).astype(np.float32) * 0.1,
            "c1_b": np.zeros(16, dtype=np.float32),
            "c2_w": np.random.randn(32, 16, 5).astype(np.float32) * 0.1,
            "c2_b": np.zeros(32, dtype=np.float32),
            "fc1_w": np.random.randn(16, 32).astype(np.float32) * 0.1,
            "fc1_b": np.zeros(16, dtype=np.float32),
            "fc2_w": np.random.randn(2, 16).astype(np.float32) * 0.1,
            "fc2_b": np.zeros(2, dtype=np.float32),
        }

    def _conv1d(self, x, w, b, padding):
        cin, lin = x.shape
        cout, _, ksize = w.shape
        x_pad = np.pad(x, ((0, 0), (padding, padding)), mode="constant")
        lout = lin + 2 * padding - ksize + 1
        out = np.zeros((cout, lout), dtype=np.float32)
        for oc in range(cout):
            val = np.zeros(lout, dtype=np.float32)
            for ic in range(cin):
                val += np.convolve(x_pad[ic], w[oc, ic, ::-1], mode="valid")
            out[oc] = val + b[oc]
        return out

    def predict_beat(self, beat_tensor):
        x = np.array(beat_tensor, dtype=np.float32)
        if x.ndim == 2:
            x = x
        elif x.ndim == 1:
            x = x.reshape(1, -1)

        x = self._conv1d(x, self.weights["c1_w"], self.weights["c1_b"], padding=3)
        x = np.maximum(0, x)

        c, l = x.shape
        if l % 2 != 0:
            x = x[:, :-1]
        x = np.maximum(x[:, 0::2], x[:, 1::2])

        x = self._conv1d(x, self.weights["c2_w"], self.weights["c2_b"], padding=2)
        x = np.maximum(0, x)

        x = np.mean(x, axis=1)

        x = np.dot(self.weights["fc1_w"], x) + self.weights["fc1_b"]
        x = np.maximum(0, x)

        logits = np.dot(self.weights["fc2_w"], x) + self.weights["fc2_b"]

        exp_l = np.exp(logits - np.max(logits))
        probs = exp_l / np.sum(exp_l)

        prob_normal = float(probs[0])
        prob_abnormal = float(probs[1])

        is_anomaly = prob_abnormal >= self.threshold
        label = "Abnormal" if is_anomaly else "Normal"
        confidence = prob_abnormal if is_anomaly else prob_normal

        return {
            "prediction": label,
            "abnormal_prob": round(prob_abnormal, 4),
            "normal_prob": round(prob_normal, 4),
            "is_anomaly": is_anomaly,
            "confidence": round(confidence * 100.0, 1),
        }

EdgeInferenceEngine = NumPyInferenceEngine
