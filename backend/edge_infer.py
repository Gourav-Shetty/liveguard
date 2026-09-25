import os
import numpy as np
from backend import config

# Expected keys/shapes for the exported Stage-1 CNN weights.
EXPECTED_WEIGHT_KEYS = {
    "c1_w": (16, 1, 7),
    "c1_b": (16,),
    "c2_w": (32, 16, 5),
    "c2_b": (32,),
    "fc1_w": (16, 32),
    "fc1_b": (16,),
    "fc2_w": (2, 16),
    "fc2_b": (2,),
}


class NumPyInferenceEngine:
    def __init__(self, weights_path: str = None, threshold: float = config.DEFAULT_ANOMALY_THRESHOLD):
        self.threshold = threshold
        self.weights = None
        npz_path = weights_path or str(config.BASE_DIR / "data" / "stage1_weights.npz")
        self._load_weights(npz_path)

    def _load_weights(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Inference weights file not found: {path}. "
                f"Generate it with: python training/export_weights_to_npz.py"
            )

        try:
            loaded = dict(np.load(path))
        except Exception as exc:
            raise ValueError(f"Failed to load inference weights from {path}: {exc!r}") from exc

        missing = [key for key in EXPECTED_WEIGHT_KEYS if key not in loaded]
        if missing:
            raise ValueError(
                f"Inference weights file {path} is missing expected keys: {', '.join(missing)}"
            )

        wrong_shapes = [
            f"{key}: expected {expected}, got {tuple(loaded[key].shape)}"
            for key, expected in EXPECTED_WEIGHT_KEYS.items()
            if tuple(loaded[key].shape) != expected
        ]
        if wrong_shapes:
            raise ValueError(
                f"Inference weights file {path} has wrong shapes: {'; '.join(wrong_shapes)}"
            )

        self.weights = {key: loaded[key] for key in EXPECTED_WEIGHT_KEYS}

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
            if x.shape[0] > x.shape[1]:
                x = x.T
        elif x.ndim == 1:
            x = x.reshape(1, -1)
        elif x.ndim == 3:
            x = x.squeeze()
            if x.ndim == 1:
                x = x.reshape(1, -1)
            elif x.shape[0] > x.shape[1]:
                x = x.T

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
