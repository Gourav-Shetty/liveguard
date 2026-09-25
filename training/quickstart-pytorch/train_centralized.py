"""
LiveGuard-EHMS — Centralized Baseline for Stage 1 (Binary Gate)

Trains the exact same CNN architecture used in the Federated runs on the 
fully pooled, centralized dataset. Applies the same threshold calibration 
(target recall >= 0.95) to allow for a direct 1:1 comparison against FedAvg/FedProx.
"""

import os
import copy
import csv
import json
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    average_precision_score,
    precision_recall_curve
)

# --------------------------------------------------------
# Configuration
# --------------------------------------------------------

# Repo-relative default (training/quickstart-pytorch/train_centralized.py ->
# repo root is parents[2]); LIVEGUARD_DATA_DIR overrides it.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATA_DIR = _REPO_ROOT / "data"


def _find_data_dir():
    if "LIVEGUARD_DATA_DIR" in os.environ and os.path.exists(os.environ["LIVEGUARD_DATA_DIR"]):
        return os.environ["LIVEGUARD_DATA_DIR"]
    candidates = [
        str(_DEFAULT_DATA_DIR),
        os.path.join(os.path.dirname(__file__), "..", "..", "data"),
        os.path.dirname(__file__),
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "mitbih_train_ready.npz")):
            return c
    return str(_DEFAULT_DATA_DIR)

DATA_DIR = _find_data_dir()

TRAIN_FILE = os.path.join(DATA_DIR, "mitbih_train_ready.npz")
VAL_FILE = os.path.join(DATA_DIR, "mitbih_val_ready.npz")
TEST_FILE = os.path.join(DATA_DIR, "mitbih_test_ready.npz")
RESULTS_CSV = os.path.join(DATA_DIR, "fl_experiment_results.csv")
MODEL_OUT = os.path.join(DATA_DIR, "stage1_cnn_centralized.pth")
HISTORY_OUT = os.path.join(DATA_DIR, "centralized_history.json")

TARGET_RECALL = 0.95
BATCH_SIZE = 128
EPOCHS = 20  # Matches the 20 rounds of FL
LEARNING_RATE = 1e-3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# --------------------------------------------------------
# Dataset & Architecture (Exact copies from task.py)
# --------------------------------------------------------

class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx].permute(1, 0), self.y[idx]

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
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
    # Convert to binary: 0 is Normal, everything else is 1 (Abnormal)
    y = (data["y"] != 0).astype(np.int64)
    return X, y

def get_abnormal_proba(model, dataloader):
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(DEVICE)
            outputs = model(X)
            proba = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(proba)
            all_targets.extend(y.numpy())
    return np.array(all_targets), np.array(all_probs)

def append_result_row(row: dict):
    file_exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# --------------------------------------------------------
# Main Execution
# --------------------------------------------------------

def main():
    print("Loading centralized data...")
    X_train, y_train = load_binary_split(TRAIN_FILE)
    X_val, y_val = load_binary_split(VAL_FILE)
    X_test, y_test = load_binary_split(TEST_FILE)

    train_loader = DataLoader(ECGDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ECGDataset(X_val, y_val), batch_size=BATCH_SIZE)
    test_loader = DataLoader(ECGDataset(X_test, y_test), batch_size=BATCH_SIZE)

    # Compute global class weights
    present = np.unique(y_train)
    counts = np.array([(y_train == c).sum() for c in present], dtype=np.float64)
    freq = counts / counts.sum()
    raw = 1.0 / (freq + 1e-6)
    raw = raw / raw.min()
    w = np.ones(2)
    for c, val in zip(present, raw):
        w[c] = val
    class_weights = torch.tensor(np.clip(w, None, 10.0), dtype=torch.float32).to(DEVICE)

    model = Net().to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n--- Starting Centralized Training ---")
    best_val_loss = np.inf
    best_model = copy.deepcopy(model.state_dict())

    # Per-epoch history persisted to centralized_history.json after training.
    history = {
        "epochs": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
    }

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        epoch_preds, epoch_targets = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item() * X.size(0)
                # Metrics only (no effect on weights/RNG): argmax decision.
                epoch_preds.append(outputs.argmax(dim=1).cpu())
                epoch_targets.append(y.cpu())
        val_loss /= len(val_loader.dataset)

        epoch_preds = torch.cat(epoch_preds).numpy()
        epoch_targets = torch.cat(epoch_targets).numpy()
        history["epochs"].append(epoch + 1)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(float(accuracy_score(epoch_targets, epoch_preds)))
        history["val_precision"].append(
            float(precision_score(epoch_targets, epoch_preds, zero_division=0))
        )
        history["val_recall"].append(
            float(recall_score(epoch_targets, epoch_preds, zero_division=0))
        )
        history["val_f1"].append(float(f1_score(epoch_targets, epoch_preds, zero_division=0)))

        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())

    print("Training complete. Loading best model for evaluation...")
    model.load_state_dict(best_model)
    torch.save(model.state_dict(), MODEL_OUT)

    # --------------------------------------------------------
    # Threshold Calibration & Evaluation
    # --------------------------------------------------------
    print(f"\n=== Threshold Optimization (Validation) — target recall >= {TARGET_RECALL:.2f} ===")
    
    val_targets, val_proba = get_abnormal_proba(model, val_loader)
    val_pr_auc = average_precision_score(val_targets, val_proba)
    print(f"Validation PR-AUC: {val_pr_auc:.4f}")

    precisions, recalls, thresholds = precision_recall_curve(val_targets, val_proba)
    valid_indices = np.where(recalls >= TARGET_RECALL)[0]

    if len(valid_indices) > 0:
        best_idx = valid_indices[np.argmax(precisions[valid_indices])]
        chosen_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1] 
    else:
        print(f"[WARNING] No threshold reached {TARGET_RECALL:.2f} recall; falling back to 0.05.")
        chosen_threshold = 0.05

    print(f"Optimal operating threshold found: {chosen_threshold:.4f}")

    val_auc = roc_auc_score(val_targets, val_proba)
    
    test_targets, test_proba = get_abnormal_proba(model, test_loader)
    test_auc = roc_auc_score(test_targets, test_proba)
    test_pr_auc = average_precision_score(test_targets, test_proba)

    val_preds = (val_proba >= chosen_threshold).astype(np.int64)
    test_preds = (test_proba >= chosen_threshold).astype(np.int64)

    for name, proba, targets, preds in [
        ("VALIDATION", val_proba, val_targets, val_preds),
        ("TEST", test_proba, test_targets, test_preds),
    ]:
        print(f"\n=== {name} (threshold={chosen_threshold:.4f}) ===")
        print(f"ROC-AUC: {roc_auc_score(targets, proba):.4f}")
        print(f"PR-AUC: {average_precision_score(targets, proba):.4f}")
        print(classification_report(targets, preds, target_names=["Normal", "Abnormal"], zero_division=0))
        print("Confusion Matrix (rows=true, cols=pred):")
        print(confusion_matrix(targets, preds))

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_file": "stage1_cnn_centralized.pth",
        "proximal_mu": "N/A (Centralized)",
        "seed": SEED,
        "num_supernodes": "Centralized",
        "chosen_threshold": round(float(chosen_threshold), 4),
        "val_auc": round(float(val_auc), 4),
        "val_pr_auc": round(float(val_pr_auc), 4),
        "test_auc": round(float(test_auc), 4),
        "test_pr_auc": round(float(test_pr_auc), 4),
        "test_recall_abnormal": round(recall_score(test_targets, test_preds, zero_division=0), 4),
        "test_precision_abnormal": round(precision_score(test_targets, test_preds, zero_division=0), 4),
    }
    append_result_row(row)
    print(f"\nAppended centralized result row to: {RESULTS_CSV}")

    # --------------------------------------------------------
    # Training history persistence (consumed by the plot script)
    # --------------------------------------------------------
    history["test"] = {
        "accuracy": float(accuracy_score(test_targets, test_preds)),
        "precision": float(precision_score(test_targets, test_preds, zero_division=0)),
        "recall": float(recall_score(test_targets, test_preds, zero_division=0)),
        "f1": float(f1_score(test_targets, test_preds, zero_division=0)),
        "auc": float(test_auc),
    }
    history["val_confusion_matrix"] = confusion_matrix(
        val_targets, val_preds, labels=[0, 1]
    ).tolist()
    history["test_confusion_matrix"] = confusion_matrix(
        test_targets, test_preds, labels=[0, 1]
    ).tolist()
    history["model_file"] = os.path.basename(MODEL_OUT)
    history["timestamp"] = datetime.now(timezone.utc).isoformat()

    with open(HISTORY_OUT, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print("[Saved] centralized_history.json")

if __name__ == "__main__":
    main()