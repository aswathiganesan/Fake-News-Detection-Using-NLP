import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/processed_news.csv"
MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "best_fake_news_model.pkl"
)

TFIDF_PATH = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.pkl"
)


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("FAKE NEWS DETECTION - MODEL TRAINING")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# CHECK LABELS
# ============================================================

print("\nLabel distribution:")

print(
    df["label"].value_counts()
)


# ============================================================
# REMOVE MISSING VALUES
# ============================================================

df["combined_text"] = (
    df["combined_text"]
    .fillna("")
    .astype(str)
)

df["label"] = (
    df["label"]
    .fillna("")
    .astype(str)
    .str.upper()
)


df = df[
    df["label"].isin(
        ["FAKE", "REAL"]
    )
]


df = df[
    df["combined_text"].str.strip() != ""
]


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df["combined_text"]

y = df["label"]


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTraining samples:", len(X_train))

print("Testing samples :", len(X_test))


# ============================================================
# TF-IDF
# ============================================================

tfidf = TfidfVectorizer(

    lowercase=True,

    stop_words="english",

    max_features=100000,

    ngram_range=(1, 2),

    sublinear_tf=True
)


# ============================================================
# LINEAR SVM
# ============================================================

svm = LinearSVC(
    C=1.0
)


# ============================================================
# CALIBRATED SVM
# ============================================================

calibrated_svm = CalibratedClassifierCV(

    svm,

    cv=5,

    method="sigmoid"
)


# ============================================================
# PIPELINE
# ============================================================

model = Pipeline(

    [

        (
            "tfidf",
            tfidf
        ),

        (
            "classifier",
            calibrated_svm
        )

    ]

)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining Calibrated Linear SVM...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(
    X_test
)

y_probability = model.predict_proba(
    X_test
)


# ============================================================
# GET CLASS ORDER
# ============================================================

classes = model.classes_

print("\nModel classes:")

print(classes)


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

real_index = list(
    classes
).index("REAL")


real_probabilities = (
    y_probability[:, real_index]
)


roc_auc = roc_auc_score(

    (y_test == "REAL").astype(int),

    real_probabilities
)


# ============================================================
# DISPLAY PERFORMANCE
# ============================================================

print("\n")
print("=" * 60)

print("MODEL PERFORMANCE")

print("=" * 60)

print(
    f"\nAccuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1-Score  : {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 60)

print("CLASSIFICATION REPORT")

print("=" * 60)

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n")
print("=" * 60)

print("CONFUSION MATRIX")

print("=" * 60)

cm = confusion_matrix(

    y_test,

    y_pred,

    labels=[
        "FAKE",
        "REAL"
    ]

)

print()

print(
    "              Predicted"
)

print(
    "              FAKE       REAL"
)

print(
    f"Actual FAKE  {cm[0][0]:8d} {cm[0][1]:10d}"
)

print(
    f"Actual REAL  {cm[1][0]:8d} {cm[1][1]:10d}"
)


# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving model...")

joblib.dump(
    model,
    MODEL_PATH
)

print(
    "Model saved to:",
    MODEL_PATH
)


# ============================================================
# SAVE TF-IDF
# ============================================================

print("\nSaving TF-IDF vectorizer...")

trained_vectorizer = (
    model.named_steps["tfidf"]
)

joblib.dump(
    trained_vectorizer,
    TFIDF_PATH
)

print(
    "TF-IDF saved to:",
    TFIDF_PATH
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 60)

print("MODEL TRAINING COMPLETED")

print("=" * 60)

print("\nSaved files:")

print(
    MODEL_PATH
)

print(
    TFIDF_PATH
)

print("\nYou can now use this model for prediction.")