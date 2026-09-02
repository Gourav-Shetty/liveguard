"""task.py -- Flower quickstart-pytorch, adapted for Stage 1 (Normal vs Abnormal gate).

Keeps the same function/class names the quickstart's client_app.py and
server_app.py already import (Net, load_data, train, test, get_weights,
set_weights), so those two files should need no changes.
"""

import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# NOTE: you're running inside WSL Ubuntu now, so the old Windows path
# (C:\Users\mirza\...) won't resolve. Either:
#   (a) point this at the Windows drive via the WSL mount, e.g.
#       "/mnt/c/Users/mirza/Downloads/livegaurd", or
#   (b) copy the mitbih_*_ready.npz files into the WSL filesystem
#       (e.g. ~/Projects/livgaurd-EHMS/data/) and point here instead.
DATA_DIR = os.environ.get("LIVEGUARD_DATA_DIR", "/mnt/c/Users/mirza/Downloads/livegaurd")
TRAIN_FILE = os.path.join(DATA_DIR, "mitbih_train_ready.npz")


class Net(nn.Module):
    """Same architecture as SmallConv1DGate in stage1_train_nn.py."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class ECGDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx].permute(1, 0), self.y[idx]  # (180,1) -> (1,180)


# ---- Partition data by PATIENT, not randomly -- this is what makes the
# simulation realistically non-IID, matching how the data would actually
# be distributed across real client sites/hospitals.
_PATIENT_CACHE = None


def _load_patient_partitions():
    global _PATIENT_CACHE
    if _PATIENT_CACHE is None:
        data = np.load(TRAIN_FILE, allow_pickle=True)
        X, y, record_id = data["X_cnn"], data["y"], data["record_id"]
        y_binary = (y != 0).astype(np.int64)
        patients = sorted(set(record_id))
        _PATIENT_CACHE = {
            pid: (X[record_id == pid], y_binary[record_id == pid]) for pid in patients
        }
    return _PATIENT_CACHE


def load_data(partition_id: int, num_partitions: int, batch_size: int = 32):
    """Return this client's local train/val DataLoaders.

    38 training patients are available. If num_partitions == 38, it's a
    clean 1:1 mapping (one client per patient -- the realistic case). If
    you run fewer virtual clients, patients are grouped round-robin.
    """
    partitions = _load_patient_partitions()
    patients = list(partitions.keys())

    my_patients = patients[partition_id::num_partitions]
    X_parts = [partitions[pid][0] for pid in my_patients]
    y_parts = [partitions[pid][1] for pid in my_patients]
    X_local = np.concatenate(X_parts)
    y_local = np.concatenate(y_parts)

    n_val = max(1, int(0.1 * len(y_local)))
    trainloader = DataLoader(
        ECGDataset(X_local[:-n_val], y_local[:-n_val]), batch_size=batch_size, shuffle=True
    )
    valloader = DataLoader(ECGDataset(X_local[-n_val:], y_local[-n_val:]), batch_size=batch_size)
    return trainloader, valloader


def _class_weights(y, max_weight=10.0):
    """Balanced weights from THIS client's local labels only -- some
    patients are almost entirely Normal, others mostly Abnormal, so a
    single global weight wouldn't fit every client."""
    present = np.unique(y)
    counts = np.array([(y == c).sum() for c in present], dtype=np.float64)
    freq = counts / counts.sum()
    raw = 1.0 / (freq + 1e-6)
    raw = raw / raw.min()
    w = np.ones(2)
    for c, val in zip(present, raw):
        w[c] = val
    return torch.tensor(np.clip(w, None, max_weight), dtype=torch.float32)


def train(net, trainloader, epochs, lr, device, proximal_mu: float = 0.0):
    """Local training step. If proximal_mu > 0, adds the FedProx proximal
    term (mu/2 * ||local_weights - global_weights||^2) to the loss, which
    penalizes local models drifting too far from the global model --
    aimed at exactly the client heterogeneity problem we saw with FedAvg
    (patients with wildly different class balance pulling the global
    model in different directions each round).
    """
    net.to(device)

    # Snapshot global weights BEFORE any local update, for the proximal term.
    global_params = [p.detach().clone() for p in net.parameters()] if proximal_mu > 0 else None

    all_y = trainloader.dataset.y.numpy()
    criterion = nn.CrossEntropyLoss(weight=_class_weights(all_y).to(device))
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for X, y in trainloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = net(X)
            loss = criterion(outputs, y)

            if proximal_mu > 0:
                prox_term = 0.0
                for local_p, global_p in zip(net.parameters(), global_params):
                    prox_term += (local_p - global_p).norm(2) ** 2
                loss = loss + (proximal_mu / 2) * prox_term

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    return running_loss / (len(trainloader) * epochs)


def test(net, testloader, device):
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    net.eval()

    correct, loss_total, total = 0, 0.0, 0
    with torch.no_grad():
        for X, y in testloader:
            X, y = X.to(device), y.to(device)
            outputs = net(X)
            loss_total += criterion(outputs, y).item()
            correct += (outputs.argmax(1) == y).sum().item()
            total += y.size(0)

    return loss_total / len(testloader), correct / total


def load_centralized_dataset(batch_size: int = 64):
    """Load the full validation set for round-by-round centralized evaluation.

    Deliberately uses mitbih_val_ready.npz, not the held-out test set --
    the test set stays completely untouched until inference_pipeline.py's
    final evaluation, so centralized monitoring during training doesn't
    leak into it.
    """
    val_file = os.path.join(DATA_DIR, "mitbih_val_ready.npz")
    data = np.load(val_file, allow_pickle=True)
    X, y = data["X_cnn"], (data["y"] != 0).astype(np.int64)
    return DataLoader(ECGDataset(X, y), batch_size=batch_size)


def set_seed(seed: int):
    """Call once at the start of server_app.py's main() for reproducibility
    across a multi-seed experiment sweep."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)