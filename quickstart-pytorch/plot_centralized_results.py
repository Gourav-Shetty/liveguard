import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. TRAINING / VALIDATION LOSS
# ============================================================

epochs = np.arange(1, 21)

train_loss = [
    0.3880, 0.2945, 0.2736, 0.2528, 0.2377,
    0.2244, 0.2166, 0.2096, 0.2060, 0.1985,
    0.1967, 0.1904, 0.1916, 0.1826, 0.1843,
    0.1777, 0.1779, 0.1761, 0.1762, 0.1702
]

val_loss = [
    0.6924, 0.5829, 0.5023, 0.6487, 0.6051,
    0.4027, 0.3997, 0.3941, 0.3724, 0.8936,
    0.4669, 0.7182, 0.3938, 0.6019, 0.4661,
    0.4083, 0.3801, 0.4254, 0.3950, 0.4217
]

plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, marker="o", label="Training Loss")
plt.plot(epochs, val_loss, marker="o", label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Centralized Training and Validation Loss")
plt.xticks(epochs)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("centralized_loss_curve.png", dpi=300)
plt.show()


# ============================================================
# 2. PERFORMANCE COMPARISON
# ============================================================

metrics = [
    "ROC-AUC",
    "PR-AUC",
    "Accuracy",
    "Abnormal\nPrecision",
    "Abnormal\nRecall",
    "Abnormal\nF1"
]

validation = [
    0.9290,
    0.8978,
    0.70,
    0.50,
    0.95,
    0.65
]

test = [
    0.8585,
    0.9036,
    0.66,
    0.62,
    0.87,
    0.72
]

x = np.arange(len(metrics))
width = 0.36

plt.figure(figsize=(11, 6))

plt.bar(x - width/2, validation, width, label="Validation")
plt.bar(x + width/2, test, width, label="Test")

plt.ylabel("Score")
plt.title("Centralized Model Performance: Validation vs Test")
plt.xticks(x, metrics)
plt.ylim(0, 1.05)
plt.grid(axis="y", alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("centralized_validation_test_performance.png", dpi=300)
plt.show()


# ============================================================
# 3. VALIDATION CONFUSION MATRIX
# ============================================================

val_cm = np.array([
    [5249, 3655],
    [189, 3601]
])

def plot_confusion_matrix(cm, title, filename):

    plt.figure(figsize=(7, 6))

    plt.imshow(cm)

    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.xticks(
        [0, 1],
        ["Normal", "Abnormal"]
    )

    plt.yticks(
        [0, 1],
        ["Normal", "Abnormal"]
    )

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                fontsize=14
            )

    plt.colorbar(label="Number of Samples")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


plot_confusion_matrix(
    val_cm,
    "Validation Confusion Matrix",
    "centralized_validation_confusion_matrix.png"
)


# ============================================================
# 4. TEST CONFUSION MATRIX
# ============================================================

test_cm = np.array([
    [2761, 3437],
    [864, 5556]
])

plot_confusion_matrix(
    test_cm,
    "Test Confusion Matrix",
    "centralized_test_confusion_matrix.png"
)


# ============================================================
# 5. CLASS-WISE METRICS
# ============================================================

classes = ["Normal", "Abnormal"]

val_precision = [0.97, 0.50]
val_recall = [0.59, 0.95]
val_f1 = [0.73, 0.65]

x = np.arange(len(classes))
width = 0.25

plt.figure(figsize=(9, 6))

plt.bar(
    x - width,
    val_precision,
    width,
    label="Precision"
)

plt.bar(
    x,
    val_recall,
    width,
    label="Recall"
)

plt.bar(
    x + width,
    val_f1,
    width,
    label="F1-score"
)

plt.ylabel("Score")
plt.xlabel("Class")
plt.title("Validation Class-wise Performance")

plt.xticks(x, classes)
plt.ylim(0, 1.05)

plt.grid(axis="y", alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("centralized_classwise_metrics.png", dpi=300)
plt.show()


print("\nGraphs generated successfully!")

print("\nFiles:")
print("1. centralized_loss_curve.png")
print("2. centralized_validation_test_performance.png")
print("3. centralized_validation_confusion_matrix.png")
print("4. centralized_test_confusion_matrix.png")
print("5. centralized_classwise_metrics.png")
