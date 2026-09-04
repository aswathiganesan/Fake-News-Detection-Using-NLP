import os
import joblib
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

METRICS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "metrics.pkl"
)

STATIC_DIR = os.path.join(
    BASE_DIR,
    "static"
)

CONFUSION_PATH = os.path.join(
    STATIC_DIR,
    "confusion_matrix.png"
)

PERFORMANCE_PATH = os.path.join(
    STATIC_DIR,
    "model_performance.png"
)


# ============================================================
# CREATE STATIC FOLDER
# ============================================================

os.makedirs(
    STATIC_DIR,
    exist_ok=True
)


# ============================================================
# LOAD METRICS
# ============================================================

if not os.path.exists(METRICS_PATH):

    raise FileNotFoundError(
        f"metrics.pkl not found:\n{METRICS_PATH}"
    )


metrics = joblib.load(
    METRICS_PATH
)


# ============================================================
# GET METRICS
# ============================================================

accuracy = float(
    metrics["accuracy"]
)

precision = float(
    metrics["precision"]
)

recall = float(
    metrics["recall"]
)

f1_score = float(
    metrics["f1_score"]
)

roc_auc = float(
    metrics["roc_auc"]
)


cm = np.array(
    metrics["confusion_matrix"]
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 60)
print("CREATING MODEL VISUALIZATIONS")
print("=" * 60)

print()
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1_score)
print("ROC-AUC  :", roc_auc)


# ============================================================
# CONFUSION MATRIX GRAPH
# ============================================================

plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Confusion Matrix"
)

plt.colorbar()


plt.xticks(
    [0, 1],
    ["FAKE", "REAL"]
)

plt.yticks(
    [0, 1],
    ["FAKE", "REAL"]
)


threshold = cm.max() / 2


for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=16,
            fontweight="bold",
            color=(
                "white"
                if cm[i, j] > threshold
                else "black"
            )
        )


plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "Actual Label"
)

plt.tight_layout()


plt.savefig(
    CONFUSION_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# MODEL PERFORMANCE GRAPH
# ============================================================

metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]


metric_values = [
    accuracy,
    precision,
    recall,
    f1_score,
    roc_auc * 100
]


plt.figure(
    figsize=(9, 6)
)


bars = plt.bar(
    metric_names,
    metric_values
)


plt.title(
    "Model Performance"
)

plt.xlabel(
    "Evaluation Metric"
)

plt.ylabel(
    "Score (%)"
)

plt.ylim(
    0,
    100
)


for bar, value in zip(
    bars,
    metric_values
):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.5,
        f"{value:.2f}%",
        ha="center",
        fontsize=10
    )


plt.tight_layout()


plt.savefig(
    PERFORMANCE_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# COMPLETED
# ============================================================

print()
print("Confusion matrix saved to:")
print(CONFUSION_PATH)

print()
print("Model performance graph saved to:")
print(PERFORMANCE_PATH)

print()
print("=" * 60)
print("VISUALIZATION COMPLETED")
print("=" * 60)