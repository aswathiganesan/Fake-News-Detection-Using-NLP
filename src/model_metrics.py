import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed_news.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_fake_news_model.pkl"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATASET_PATH)

df["combined_text"] = (
    df["combined_text"]
    .fillna("")
    .astype(str)
)

df["label"] = (
    df["label"]
    .astype(str)
    .str.upper()
    .str.strip()
)


X = df["combined_text"]
y = df["label"]


# ============================================================
# SAME TEST SPLIT USED DURING TRAINING
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    MODEL_PATH
)


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)

probabilities = model.predict_proba(
    X_test
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    pos_label="REAL"
)

recall = recall_score(
    y_test,
    y_pred,
    pos_label="REAL"
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label="REAL"
)


# ============================================================
# ROC-AUC
# ============================================================

classes = list(
    model.classes_
)

real_index = classes.index(
    "REAL"
)

real_probabilities = (
    probabilities[:, real_index]
)

roc_auc = roc_auc_score(
    (y_test == "REAL").astype(int),
    real_probabilities
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[
        "FAKE",
        "REAL"
    ]
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)

print()
print("Confusion Matrix:")
print(cm)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "accuracy": round(
        accuracy * 100,
        2
    ),

    "precision": round(
        precision * 100,
        2
    ),

    "recall": round(
        recall * 100,
        2
    ),

    "f1_score": round(
        f1 * 100,
        2
    ),

    "roc_auc": round(
        roc_auc,
        4
    ),

    "confusion_matrix": cm.tolist()

}


metrics_path = os.path.join(
    BASE_DIR,
    "models",
    "metrics.pkl"
)


joblib.dump(
    metrics,
    metrics_path
)


print()
print(
    "Metrics saved to:"
)

print(
    metrics_path
)

print("=" * 60)