"""
Run this once after training — `flwr run .` (federated) or
train_centralized.py (centralized) — and a stage1_cnn_*.pth file has been
saved in data/.

Loads the model's weights, sweeps thresholds against the validation set
(maximizing precision for TARGET_RECALL), appends a row of results to a CSV,
and writes data/stage1_threshold.json for backend/config.py to consume.
"""

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    average_precision_score,
    precision_recall_curve
)

def _find_data_dir():
    # Mirrors backend/config.py: env override first, else repo data/ dir.
    env_dir = os.environ.get("LIVEGUARD_DATA_DIR")
    if env_dir:
        return env_dir
    return str(Path(__file__).resolve().parents[2] / "data")

DATA_DIR = _find_data_dir()
VAL_FILE = os.path.join(DATA_DIR, "mitbih_val_ready.npz")
TEST_FILE = os.path.join(DATA_DIR, "mitbih_test_ready.npz")
RESULTS_CSV = os.path.join(DATA_DIR, "fl_experiment_results.csv")

TARGET_RECALL = 0.95
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32, 16), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def load_binary_split(path):
    data = np.load(path)
    X = data["X_cnn"]
    y = (data["y"] != 0).astype(np.int64)
    return X, y

def get_abnormal_proba(model, X):
    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1).to(DEVICE)
    with torch.no_grad():
        outputs = model(X_tensor)
        proba = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
    return proba

def append_result_row(row: dict):
    file_exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", default=None,
                        help="Filename (within DATA_DIR) of the saved model. "
                             "Default: stage1_cnn.pth (FL), falling back to "
                             "stage1_cnn_centralized.pth")
    parser.add_argument("--mu", type=float, default=None, help="proximal_mu used for this run, for logging")
    parser.add_argument("--seed", type=int, default=None, help="seed used for this run, for logging")
    parser.add_argument("--num-supernodes", type=int, default=None, help="num-supernodes used for this run, for logging")
    args = parser.parse_args()

    model_file = args.model_file
    if model_file is None:
        for candidate in ("stage1_cnn.pth", "stage1_cnn_centralized.pth"):
            if os.path.exists(os.path.join(DATA_DIR, candidate)):
                model_file = candidate
                break
        if model_file is None:
            raise SystemExit(
                f"No stage1 model found in {DATA_DIR}. Train first "
                f"(training/quickstart-pytorch/train_centralized.py or `flwr run .`) "
                f"or pass --model-file."
            )

    model_path = os.path.join(DATA_DIR, model_file)
    print(f"Loading model from {model_path}...")
    model = Net().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    X_val, y_val = load_binary_split(VAL_FILE)
    X_test, y_test = load_binary_split(TEST_FILE)

    val_proba = get_abnormal_proba(model, X_val)

    print(f"\n=== Threshold Optimization (Validation) — target recall >= {TARGET_RECALL:.2f} ===")
    
    val_pr_auc = average_precision_score(y_val, val_proba)
    print(f"Validation PR-AUC: {val_pr_auc:.4f}")

    precisions, recalls, thresholds = precision_recall_curve(y_val, val_proba)
    valid_indices = np.where(recalls >= TARGET_RECALL)[0]

    if len(valid_indices) > 0:
        best_idx = valid_indices[np.argmax(precisions[valid_indices])]
        if best_idx < len(thresholds):
            chosen_threshold = thresholds[best_idx]
        else:
            chosen_threshold = thresholds[-1] 
    else:
        print(f"[WARNING] No threshold reached {TARGET_RECALL:.2f} recall; falling back to 0.05.")
        chosen_threshold = 0.05

    print(f"Optimal operating threshold found: {chosen_threshold:.4f}")

    val_auc = roc_auc_score(y_val, val_proba)
    test_proba = get_abnormal_proba(model, X_test)
    test_auc = roc_auc_score(y_test, test_proba)
    test_pr_auc = average_precision_score(y_test, test_proba)

    val_preds = (val_proba >= chosen_threshold).astype(np.int64)
    test_preds = (test_proba >= chosen_threshold).astype(np.int64)

    for name, proba, targets, preds in [
        ("VALIDATION", val_proba, y_val, val_preds),
        ("TEST", test_proba, y_test, test_preds),
    ]:
        print(f"\n=== {name} (threshold={chosen_threshold:.4f}) ===")
        print(f"ROC-AUC: {roc_auc_score(targets, proba):.4f}")
        print(f"PR-AUC: {average_precision_score(targets, proba):.4f}")
        print(classification_report(targets, preds, target_names=["Normal", "Abnormal"], zero_division=0))
        print("Confusion Matrix (rows=true, cols=pred):")
        print(confusion_matrix(targets, preds))

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_file": model_file,
        "proximal_mu": args.mu,
        "seed": args.seed,
        "num_supernodes": args.num_supernodes,
        "chosen_threshold": round(float(chosen_threshold), 4),
        "val_auc": round(float(val_auc), 4),
        "val_pr_auc": round(float(val_pr_auc), 4),
        "test_auc": round(float(test_auc), 4),
        "test_pr_auc": round(float(test_pr_auc), 4),
        "test_recall_abnormal": round(recall_score(y_test, test_preds, zero_division=0), 4),
        "test_precision_abnormal": round(precision_score(y_test, test_preds, zero_division=0), 4),
    }
    append_result_row(row)
    print(f"\nAppended result row to: {RESULTS_CSV}")

    payload = {
        "threshold": float(chosen_threshold),
        "target_recall": float(TARGET_RECALL),
        "val_pr_auc": float(val_pr_auc),
        "val_auc": float(val_auc),
        "test_auc": float(test_auc),
        "test_pr_auc": float(test_pr_auc),
        "test_precision_abnormal": float(precision_score(y_test, test_preds, zero_division=0)),
        "test_recall_abnormal": float(recall_score(y_test, test_preds, zero_division=0)),
        "model_file": os.path.basename(model_path),
        "timestamp": row["timestamp"],
    }
    threshold_json = os.path.join(DATA_DIR, "stage1_threshold.json")
    with open(threshold_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[Saved] stage1_threshold.json (threshold={payload['threshold']:.4f})")

if __name__ == "__main__":
    main()