import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("LIVEGUARD_DATA_DIR", os.path.join(_REPO_ROOT, "data"))
TRAIN_FILE = os.path.join(DATA_DIR, "mitbih_train_ready.npz")
VAL_FILE = os.path.join(DATA_DIR, "mitbih_val_ready.npz")
TEST_FILE = os.path.join(DATA_DIR, "mitbih_test_ready.npz")
MODEL_OUT = os.path.join(DATA_DIR, "stage2_multiclass_cnn.pth")

BATCH_SIZE = 256
EPOCHS = 12
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["N (Normal)", "S (Supraventricular)", "V (Ventricular)", "F (Fusion)", "Q (Paced/Unknown)"]

class ECGDataset(Dataset):
    def __init__(self, X, y):
        if X.ndim == 3 and X.shape[-1] == 1:
            X = np.ascontiguousarray(np.transpose(X, (0, 2, 1)))
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class Stage2MulticlassNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def load_data():
    train_data = np.load(TRAIN_FILE)
    val_data = np.load(VAL_FILE)
    test_data = np.load(TEST_FILE)

    train_loader = DataLoader(ECGDataset(train_data["X_cnn"], train_data["y"]), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ECGDataset(val_data["X_cnn"], val_data["y"]), batch_size=BATCH_SIZE)
    test_loader = DataLoader(ECGDataset(test_data["X_cnn"], test_data["y"]), batch_size=BATCH_SIZE)

    # Balanced class weights for handling minority arrhythmia classes
    y_train = train_data["y"]
    present = np.unique(y_train)
    counts = np.array([(y_train == c).sum() for c in present], dtype=np.float64)
    freq = counts / counts.sum()
    raw = 1.0 / (freq + 1e-4)
    raw = raw / raw.min()
    w = np.ones(5)
    for c, val in zip(present, raw):
        w[c] = min(val, 25.0)
    class_weights = torch.tensor(w, dtype=torch.float32).to(DEVICE)

    return train_loader, val_loader, test_loader, class_weights

def main():
    print(f"Using device: {DEVICE}")
    print("Loading data for Stage 2 (AAMI 5-Class Arrhythmia Classification)...")
    train_loader, val_loader, test_loader, class_weights = load_data()

    model = Stage2MulticlassNet(num_classes=5).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n--- Starting Stage 2 Training ---")
    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                out = model(X)
                loss = criterion(out, y)
                val_loss += loss.item() * X.size(0)
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total

        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())

    print("\nTraining complete. Evaluating best checkpoint on held-out TEST set...")
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), MODEL_OUT)
    print(f"Saved Stage 2 Model to: {MODEL_OUT}")

    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(DEVICE)
            out = model(X)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    print("\n=== Stage 2 Test Evaluation (AAMI Standard) ===")
    print(classification_report(all_targets, all_preds, target_names=CLASS_NAMES, zero_division=0))
    print("Confusion Matrix (rows=True, cols=Pred):")
    print(confusion_matrix(all_targets, all_preds))

if __name__ == "__main__":
    main()
