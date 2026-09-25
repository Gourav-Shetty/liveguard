#!/usr/bin/env python3
"""Plot centralized training results and the FL-vs-centralized benchmark.

All inputs are read from the repo's ``data/`` directory (resolved from this
file, so the script works from any working directory):

  data/centralized_history.json  per-epoch curves + final metrics,
                                 written by train_centralized.py
  data/fl_experiment_results.csv benchmark log shared by centralized and FL runs

Usage:
  python plot_centralized_results.py [OUTPUT_DIR]

OUTPUT_DIR defaults to this script's own directory (where the tracked PNGs
live); pass a scratch directory to avoid overwriting them.

Missing or unreadable inputs are reported with a single-line message (never a
traceback) and only the figures supported by the available data are produced.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; must be set before pyplot

# Belt-and-braces: never let a narrow console encoding turn a message into a
# UnicodeEncodeError traceback (e.g. when a caller's data contains non-ASCII).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, OSError):
        pass

import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------
# Paths - everything is resolved from this file.
# training/quickstart-pytorch/plot_centralized_results.py -> parents[2]
# is the repo root.
# ------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]


def _find_data_dir() -> Path:
    """``data/`` at the repo root; LIVEGUARD_DATA_DIR overrides it
    (same convention as train_centralized.py)."""
    override = os.environ.get("LIVEGUARD_DATA_DIR")
    if override and Path(override).is_dir():
        return Path(override)
    return REPO_ROOT / "data"


DATA_DIR = _find_data_dir()
HISTORY_PATH = DATA_DIR / "centralized_history.json"
RESULTS_CSV = DATA_DIR / "fl_experiment_results.csv"

USAGE = "usage: python training/quickstart-pytorch/plot_centralized_results.py [OUTPUT_DIR]"


# ------------------------------------------------------------------
# Small data helpers
# ------------------------------------------------------------------

def _num(value):
    """Best-effort float conversion; None for bools/None/garbage."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _series(value):
    """Coerce a JSON list into a list of floats (unparseable entries -> None)."""
    if not isinstance(value, (list, tuple)):
        return []
    return [_num(item) for item in value]


def _pad(series, n):
    """Pad/truncate to length n, using NaN for holes so matplotlib skips them."""
    out = [v if v is not None else float("nan") for v in series[:n]]
    out.extend([float("nan")] * (n - len(out)))
    return out


def _last_numeric(series):
    """Final numeric value of a per-epoch series (the final-epoch metric)."""
    for value in reversed(_series(series)):
        if value is not None:
            return value
    return None


def _as_matrix(value):
    """Return a 2x2 nested list of floats, or None if the shape is wrong."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    rows = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            return None
        cells = [_num(cell) for cell in row]
        if any(cell is None for cell in cells):
            return None
        rows.append(cells)
    return rows


def _save(fig, out_dir, filename):
    path = out_dir / filename
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return filename


def _run_label(row, fallback_index):
    """Human-readable label for one fl_experiment_results.csv row."""
    supernodes = (row.get("num_supernodes") or "").strip()
    mu = (row.get("proximal_mu") or "").strip()
    seed = (row.get("seed") or "").strip()
    model = (row.get("model_file") or "").lower()

    if supernodes.lower() == "centralized" or "centralized" in model:
        label = "Centralized"
    elif supernodes:
        count = _num(supernodes)
        if count is not None and float(count).is_integer():
            description = f"{int(count)} supernodes"
        else:  # e.g. "FedAvg (4 supernodes)" - keep the wording, drop nesting
            description = supernodes.replace("(", " ").replace(")", "").strip()
        label = f"FL ({description}"
        if mu and not mu.upper().startswith("N/A"):
            label += f", mu={mu}"
        label += ")"
    else:
        label = f"Run {fallback_index + 1}"
    if seed:
        label += f" seed {seed}"
    threshold = _num(row.get("chosen_threshold"))
    if threshold is not None:
        label += f" thr={threshold:g}"
    return label


# ------------------------------------------------------------------
# Data loading (never raises, never prints a traceback)
# ------------------------------------------------------------------

def load_history():
    """centralized_history.json -> dict, or None with a one-line reason."""
    if not HISTORY_PATH.is_file():
        print(
            f"{HISTORY_PATH.name} not found - run "
            "training/quickstart-pytorch/train_centralized.py first"
        )
        return None
    try:
        with open(HISTORY_PATH, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"{HISTORY_PATH.name} could not be read ({exc}) - re-run train_centralized.py")
        return None
    if not isinstance(data, dict):
        print(f"{HISTORY_PATH.name} has an unexpected layout - re-run train_centralized.py")
        return None
    return data


def load_results_rows():
    """fl_experiment_results.csv -> list of dict rows, or None with a reason."""
    if not RESULTS_CSV.is_file():
        print(
            f"{RESULTS_CSV.name} not found - run "
            "training/quickstart-pytorch/train_centralized.py (or the FL benchmark) first"
        )
        return None
    try:
        with open(RESULTS_CSV, newline="", encoding="utf-8") as fh:
            rows = [dict(row) for row in csv.DictReader(fh)]
    except (OSError, csv.Error) as exc:
        print(f"{RESULTS_CSV.name} could not be read ({exc}) - re-run the benchmark")
        return None
    if not rows:
        print(f"{RESULTS_CSV.name} has no result rows yet - run the benchmark first")
        return None
    return rows


# ------------------------------------------------------------------
# 1. Training / validation loss  ->  centralized_loss_curve.png
# ------------------------------------------------------------------

def plot_loss_curve(history, out_dir):
    train_loss = _series(history.get("train_loss"))
    val_loss = _series(history.get("val_loss"))
    n = max(len(train_loss), len(val_loss))
    if n == 0:
        print("centralized_history.json has no train_loss/val_loss series - skipping loss curve")
        return None

    epochs = _series(history.get("epochs"))
    if len(epochs) != n:
        epochs = [float(i) for i in range(1, n + 1)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, _pad(train_loss, n), marker="o", label="Training Loss")
    ax.plot(epochs, _pad(val_loss, n), marker="o", label="Validation Loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Centralized Training and Validation Loss")
    if n <= 40:
        ax.set_xticks(epochs)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return _save(fig, out_dir, "centralized_loss_curve.png")


# ------------------------------------------------------------------
# 2. Validation vs test performance -> centralized_validation_test_performance.png
#    Validation bars come from the final epoch of the history file; test bars
#    from its `test` block. AUC / PR-AUC (and test precision/recall as a
#    fallback) are filled in from the centralized row of the results CSV.
# ------------------------------------------------------------------

_METRICS = [
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
    ("accuracy", "Accuracy"),
    ("precision", "Abnormal\nPrecision"),
    ("recall", "Abnormal\nRecall"),
    ("f1", "Abnormal\nF1"),
]


def _collect_performance(history, rows):
    val, test = {}, {}
    if history:
        test_block = history.get("test") if isinstance(history.get("test"), dict) else {}
        for key in ("accuracy", "precision", "recall", "f1"):
            value = _num(test_block.get(key))
            if value is not None:
                test[key] = value
        auc = _num(test_block.get("auc"))
        if auc is not None:
            test["roc_auc"] = auc

        for key, column in (
            ("accuracy", "val_accuracy"),
            ("precision", "val_precision"),
            ("recall", "val_recall"),
            ("f1", "val_f1"),
        ):
            value = _last_numeric(history.get(column))
            if value is not None:
                val[key] = value

    if rows:
        centralized = next(
            (
                row
                for row in rows
                if (row.get("num_supernodes") or "").lower() == "centralized"
                or "centralized" in (row.get("model_file") or "").lower()
            ),
            rows[-1],
        )
        for key, column in (("roc_auc", "val_auc"), ("pr_auc", "val_pr_auc")):
            val.setdefault(key, _num(centralized.get(column)))
        for key, column in (
            ("roc_auc", "test_auc"),
            ("pr_auc", "test_pr_auc"),
            ("precision", "test_precision_abnormal"),
            ("recall", "test_recall_abnormal"),
        ):
            test.setdefault(key, _num(centralized.get(column)))
    return val, test


def plot_performance(history, rows, out_dir):
    val, test = _collect_performance(history, rows)
    metrics = [(key, label) for key, label in _METRICS
               if val.get(key) is not None or test.get(key) is not None]
    if not metrics:
        print("no validation/test metrics available - skipping performance comparison")
        return None

    x = np.arange(len(metrics))
    width = 0.36
    val_values = [val.get(key, float("nan")) for key, _ in metrics]
    test_values = [test.get(key, float("nan")) for key, _ in metrics]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width / 2, val_values, width, label="Validation")
    ax.bar(x + width / 2, test_values, width, label="Test")

    ax.set_ylabel("Score")
    ax.set_title("Centralized Model Performance: Validation vs Test")
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return _save(fig, out_dir, "centralized_validation_test_performance.png")


# ------------------------------------------------------------------
# 3./4. Confusion matrices (rows = true label, cols = predicted label)
# ------------------------------------------------------------------

def plot_confusion_matrix(cm, title, out_dir, filename):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(np.asarray(cm, dtype=float))

    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_xticks([0, 1], ["Normal", "Abnormal"])
    ax.set_yticks([0, 1], ["Normal", "Abnormal"])

    for i in range(2):
        for j in range(2):
            value = cm[i][j]
            text = f"{int(round(value)):,}" if float(value).is_integer() else f"{value:,.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=14)

    plt.colorbar(ax.images[0], ax=ax, label="Number of Samples")

    fig.tight_layout()
    return _save(fig, out_dir, filename)


def plot_validation_cm(history, out_dir):
    cm = _as_matrix(history.get("val_confusion_matrix"))
    if cm is None:
        print("val_confusion_matrix missing from centralized_history.json - skipping validation CM")
        return None
    return plot_confusion_matrix(
        cm, "Validation Confusion Matrix", out_dir,
        "centralized_validation_confusion_matrix.png",
    )


def plot_test_cm(history, out_dir):
    cm = _as_matrix(history.get("test_confusion_matrix"))
    if cm is None:
        print("test_confusion_matrix missing from centralized_history.json - skipping test CM")
        return None
    return plot_confusion_matrix(
        cm, "Test Confusion Matrix", out_dir,
        "centralized_test_confusion_matrix.png",
    )


# ------------------------------------------------------------------
# 5. Class-wise metrics -> centralized_classwise_metrics.png
#    Per-class precision/recall/F1 are derived from the validation confusion
#    matrix: rows = true class, cols = predicted class, [[tn, fp], [fn, tp]].
# ------------------------------------------------------------------

def _prf(tp_like, fp, fn):
    precision = tp_like / (tp_like + fp) if (tp_like + fp) else 0.0
    recall = tp_like / (tp_like + fn) if (tp_like + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def plot_classwise(history, out_dir):
    cm = _as_matrix(history.get("val_confusion_matrix"))
    if cm is None:
        print("val_confusion_matrix missing from centralized_history.json - skipping class-wise metrics")
        return None
    (tn, fp), (fn, tp) = cm

    classes = ["Normal", "Abnormal"]
    precision, recall, f1 = [], [], []
    for p, r, f in (_prf(tn, fn, fp), _prf(tp, fp, fn)):
        precision.append(p)
        recall.append(r)
        f1.append(f)

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x, recall, width, label="Recall")
    ax.bar(x + width, f1, width, label="F1-score")

    ax.set_ylabel("Score")
    ax.set_xlabel("Class")
    ax.set_title("Validation Class-wise Performance")
    ax.set_xticks(x, classes)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return _save(fig, out_dir, "centralized_classwise_metrics.png")


# ------------------------------------------------------------------
# 6. FL vs centralized benchmark -> fl_vs_centralized.png
#    One bar group per metric column of fl_experiment_results.csv, one bar
#    per result row (centralized and FL runs side by side).
# ------------------------------------------------------------------

_FL_METRICS = [
    ("val_auc", "Validation\nROC-AUC"),
    ("val_pr_auc", "Validation\nPR-AUC"),
    ("test_auc", "Test\nROC-AUC"),
    ("test_pr_auc", "Test\nPR-AUC"),
    ("test_recall_abnormal", "Test Abnormal\nRecall"),
    ("test_precision_abnormal", "Test Abnormal\nPrecision"),
]


def plot_fl_comparison(rows, out_dir):
    usable = [
        row for row in rows
        if any(_num(row.get(column)) is not None for column, _ in _FL_METRICS)
    ]
    if not usable:
        print(f"{RESULTS_CSV.name} rows contain no known metrics - skipping FL comparison")
        return None

    fl_rows = [
        row for row in usable
        if (row.get("num_supernodes") or "").strip().lower() not in ("", "centralized")
        and "centralized" not in (row.get("model_file") or "").lower()
    ]
    if not fl_rows:
        print(f"{RESULTS_CSV.name} has no FL rows yet - plotting centralized row only")

    x = np.arange(len(_FL_METRICS))
    width = min(0.36, 0.8 / len(usable))

    fig, ax = plt.subplots(figsize=(12, 6))
    seen = {}
    for index, row in enumerate(usable):
        label = _run_label(row, index)
        seen[label] = seen.get(label, 0) + 1
        if seen[label] > 1:  # keep legend entries unique
            label = f"{label} [{seen[label]}]"
        values = [_num(row.get(column)) for column, _ in _FL_METRICS]
        values = [v if v is not None else float("nan") for v in values]
        offset = (index - (len(usable) - 1) / 2) * width
        ax.bar(x + offset, values, width, label=label)

    ax.set_ylabel("Score")
    ax.set_title("Centralized vs Federated Learning (data/fl_experiment_results.csv)")
    ax.set_xticks(x, [label for _, label in _FL_METRICS])
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    return _save(fig, out_dir, "fl_vs_centralized.png")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0

    out_dir = Path(argv[0]) if argv else SCRIPT_DIR
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"cannot create output directory {out_dir}: {exc}")
        return 1

    history = load_history()
    rows = load_results_rows()
    if history is None and rows is None:
        print("no usable data in data/ - nothing to plot")
        return 1

    jobs = []
    if history is not None:
        jobs += [
            lambda: plot_loss_curve(history, out_dir),
            lambda: plot_performance(history, rows, out_dir),
            lambda: plot_validation_cm(history, out_dir),
            lambda: plot_test_cm(history, out_dir),
            lambda: plot_classwise(history, out_dir),
        ]
    if rows is not None:
        jobs.append(lambda: plot_fl_comparison(rows, out_dir))

    produced = []
    for job in jobs:
        try:
            filename = job()
        except Exception as exc:  # malformed-but-present data: message, no traceback
            print(f"plot skipped ({type(exc).__name__}: {exc})")
            continue
        if filename:
            produced.append(filename)

    if not produced:
        print("no plots could be produced from the available data")
        return 1

    print("\nGraphs generated successfully!")
    print(f"\nOutput directory: {out_dir}")
    print("\nFiles:")
    for index, filename in enumerate(produced, start=1):
        print(f"{index}. {filename}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
