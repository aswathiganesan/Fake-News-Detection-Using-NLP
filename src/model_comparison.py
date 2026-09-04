import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
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

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "model_comparison.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(
    DATASET_PATH
)


# ============================================================
# CLEAN DATA
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
    .str.strip()
)

df = df[
    df["label"].isin(
        ["FAKE", "REAL"]
    )
]

df = df[
    df["combined_text"].str.strip() != ""
]


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


print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            C=1.0
        ),

    "Naive Bayes":
        MultinomialNB(),

    "Linear SVM":
        CalibratedClassifierCV(
            LinearSVC(
                C=1.0
            ),
            cv=5,
            method="sigmoid"
        )

}


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# TRAIN EACH MODEL
# ============================================================

for name, classifier in models.items():

    print()
    print("-" * 60)
    print(
        "Training:",
        name
    )
    print("-" * 60)


    pipeline = Pipeline(

        [

            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    max_features=100000,
                    ngram_range=(1, 2),
                    sublinear_tf=True
                )
            ),

            (
                "classifier",
                classifier
            )

        ]

    )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    pipeline.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    y_pred = pipeline.predict(
        X_test
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results.append(

        {

            "Model": name,

            "Accuracy": accuracy,

            "Precision": precision,

            "Recall": recall,

            "F1_Score": f1

        }

    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# BEST MODEL
# ============================================================

best_index = results_df[
    "F1_Score"
].idxmax()

best_model = results_df.loc[
    best_index,
    "Model"
]


# ============================================================
# SAVE CSV
# ============================================================

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print()
print("=" * 60)
print("MODEL COMPARISON RESULTS")
print("=" * 60)

print()

print(
    results_df.to_string(
        index=False
    )
)

print()

print(
    "Best Model:",
    best_model
)

print()

print(
    "Results saved to:"
)

print(
    OUTPUT_PATH
)

print("=" * 60)