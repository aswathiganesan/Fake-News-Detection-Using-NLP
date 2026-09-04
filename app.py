from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
import numpy as np


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_fake_news_model.pkl"
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed_news.csv"
)

METRICS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "metrics.pkl"
)

COMPARISON_PATH = os.path.join(
    BASE_DIR,
    "data",
    "model_comparison.csv"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("FAKE NEWS DETECTION SYSTEM")
print("=" * 60)

print("\nLoading trained model...")

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

model = joblib.load(
    MODEL_PATH
)

print("Model loaded successfully.")

print(
    "Model classes:",
    model.classes_
)


# ============================================================
# GET TF-IDF VECTORIZER
# ============================================================

if hasattr(model, "named_steps"):

    vectorizer = model.named_steps.get(
        "tfidf"
    )

    classifier = model.named_steps.get(
        "classifier"
    )

else:

    vectorizer = None

    classifier = model


# ============================================================
# TEXT STATISTICS
# ============================================================

def get_text_statistics(text):

    words = text.split()

    word_count = len(words)

    character_count = len(text)

    cleaned_words = []

    for word in words:

        cleaned = word.lower().strip(
            ".,!?;:\"'()[]{}"
        )

        if cleaned:

            cleaned_words.append(
                cleaned
            )

    unique_word_count = len(
        set(cleaned_words)
    )

    return (
        word_count,
        character_count,
        unique_word_count
    )


# ============================================================
# TF-IDF FEATURE COUNT
# ============================================================

def get_tfidf_feature_count(text):

    if vectorizer is None:

        return 0

    try:

        transformed = vectorizer.transform(
            [text]
        )

        values = transformed.toarray()[0]

        return int(
            np.sum(values > 0)
        )

    except Exception as error:

        print(
            "TF-IDF error:",
            error
        )

        return 0


# ============================================================
# GET MODEL COEFFICIENTS
# ============================================================

def get_coefficients():

    if classifier is None:

        return None


    # --------------------------------------------------------
    # Direct classifier
    # --------------------------------------------------------

    if hasattr(
        classifier,
        "coef_"
    ):

        try:

            return classifier.coef_[0]

        except Exception:

            pass


    # --------------------------------------------------------
    # Calibrated classifier
    # --------------------------------------------------------

    if hasattr(
        classifier,
        "calibrated_classifiers_"
    ):

        all_coefficients = []


        for calibrated in (
            classifier
            .calibrated_classifiers_
        ):

            estimator = None


            if hasattr(
                calibrated,
                "estimator"
            ):

                estimator = (
                    calibrated.estimator
                )


            elif hasattr(
                calibrated,
                "base_estimator"
            ):

                estimator = (
                    calibrated.base_estimator
                )


            if estimator is None:

                continue


            if hasattr(
                estimator,
                "coef_"
            ):

                try:

                    all_coefficients.append(
                        estimator.coef_[0]
                    )

                except Exception:

                    pass


        if all_coefficients:

            return np.mean(
                all_coefficients,
                axis=0
            )


    return None


# ============================================================
# EXPLAIN PREDICTION
# ============================================================

def explain_prediction(
    text,
    top_n=5
):

    fake_features = []

    real_features = []


    if vectorizer is None:

        return (
            fake_features,
            real_features
        )


    try:

        X = vectorizer.transform(
            [text]
        )

        tfidf_values = X.toarray()[0]

        feature_names = (
            vectorizer
            .get_feature_names_out()
        )

        coefficients = (
            get_coefficients()
        )


        if coefficients is None:

            return (
                fake_features,
                real_features
            )


        contributions = []


        limit = min(
            len(feature_names),
            len(tfidf_values),
            len(coefficients)
        )


        for i in range(limit):

            if tfidf_values[i] <= 0:

                continue


            contribution = (
                tfidf_values[i]
                * coefficients[i]
            )


            contributions.append(
                (
                    feature_names[i],
                    float(contribution)
                )
            )


        # ----------------------------------------------------
        # FAKE-related
        # ----------------------------------------------------

        fake_sorted = sorted(
            [
                item
                for item in contributions
                if item[1] > 0
            ],
            key=lambda item: abs(item[1]),
            reverse=True
        )


        # ----------------------------------------------------
        # REAL-related
        # ----------------------------------------------------

        real_sorted = sorted(
            [
                item
                for item in contributions
                if item[1] < 0
            ],
            key=lambda item: abs(item[1]),
            reverse=True
        )


        fake_features = [

            (
                word,
                round(
                    abs(value),
                    4
                )
            )

            for word, value
            in fake_sorted[:top_n]

        ]


        real_features = [

            (
                word,
                round(
                    abs(value),
                    4
                )
            )

            for word, value
            in real_sorted[:top_n]

        ]


    except Exception as error:

        print(
            "Explanation error:",
            error
        )


    return (
        fake_features,
        real_features
    )


# ============================================================
# RISK LEVEL
# ============================================================

def calculate_risk(
    prediction,
    confidence
):

    if prediction == "FAKE":

        if confidence >= 90:

            return "HIGH"

        elif confidence >= 70:

            return "MEDIUM"

        else:

            return "LOW"


    if prediction == "REAL":

        if confidence >= 90:

            return "LOW"

        elif confidence >= 70:

            return "MEDIUM"

        else:

            return "HIGH"


    return "UNKNOWN"


# ============================================================
# CREDIBILITY SCORE
# ============================================================

def calculate_credibility(
    prediction,
    confidence
):

    if prediction == "REAL":

        score = confidence

    else:

        score = 100 - confidence


    score = max(
        0,
        min(
            100,
            score
        )
    )


    return round(
        score,
        2
    )


# ============================================================
# DETECTION PAGE
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def home():

    prediction = None

    confidence = None

    fake_probability = None

    real_probability = None

    risk_level = None

    credibility_score = None

    news_text = ""

    fake_features = []

    real_features = []

    word_count = None

    character_count = None

    unique_word_count = None

    tfidf_feature_count = None


    if request.method == "POST":

        news_text = request.form.get(
            "news",
            ""
        ).strip()


        if news_text:

            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            prediction = model.predict(
                [news_text]
            )[0]

            prediction = str(
                prediction
            ).upper()


            # ------------------------------------------------
            # Probability
            # ------------------------------------------------

            try:

                probabilities = (
                    model.predict_proba(
                        [news_text]
                    )[0]
                )

                classes = list(
                    model.classes_
                )


                probability_dict = dict(
                    zip(
                        classes,
                        probabilities
                    )
                )


                fake_probability = round(
                    probability_dict.get(
                        "FAKE",
                        0
                    ) * 100,
                    2
                )


                real_probability = round(
                    probability_dict.get(
                        "REAL",
                        0
                    ) * 100,
                    2
                )


                confidence = round(
                    max(probabilities) * 100,
                    2
                )


            except Exception as error:

                print(
                    "Probability error:",
                    error
                )


            # ------------------------------------------------
            # Risk and credibility
            # ------------------------------------------------

            if confidence is not None:

                risk_level = calculate_risk(
                    prediction,
                    confidence
                )


                credibility_score = (
                    calculate_credibility(
                        prediction,
                        confidence
                    )
                )


            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            (
                word_count,
                character_count,
                unique_word_count

            ) = get_text_statistics(
                news_text
            )


            tfidf_feature_count = (
                get_tfidf_feature_count(
                    news_text
                )
            )


            # ------------------------------------------------
            # Explainability
            # ------------------------------------------------

            (
                fake_features,
                real_features

            ) = explain_prediction(
                news_text
            )


            # ------------------------------------------------
            # Terminal output
            # ------------------------------------------------

            print()
            print("-" * 60)
            print("NEWS ANALYSIS")
            print("-" * 60)

            print(
                "Prediction:",
                prediction
            )

            print(
                "Confidence:",
                confidence,
                "%"
            )

            print(
                "REAL probability:",
                real_probability,
                "%"
            )

            print(
                "FAKE probability:",
                fake_probability,
                "%"
            )

            print(
                "Risk:",
                risk_level
            )

            print(
                "Credibility:",
                credibility_score
            )

            print("-" * 60)


    return render_template(

        "index.html",

        prediction=prediction,

        confidence=confidence,

        fake_probability=fake_probability,

        real_probability=real_probability,

        risk_level=risk_level,

        credibility_score=credibility_score,

        news_text=news_text,

        fake_features=fake_features,

        real_features=real_features,

        word_count=word_count,

        character_count=character_count,

        unique_word_count=unique_word_count,

        tfidf_feature_count=tfidf_feature_count

    )


# ============================================================
# DASHBOARD PAGE
# ============================================================

@app.route("/dashboard")
def dashboard():

    if not os.path.exists(
        DATASET_PATH
    ):

        return (
            "processed_news.csv not found. "
            "Run prepare_data.py first.",
            404
        )


    df = pd.read_csv(
        DATASET_PATH
    )


    df["label"] = (
        df["label"]
        .astype(str)
        .str.upper()
        .str.strip()
    )


    total_articles = len(df)


    fake_count = int(
        (
            df["label"] == "FAKE"
        ).sum()
    )


    real_count = int(
        (
            df["label"] == "REAL"
        ).sum()
    )


    if total_articles > 0:

        fake_percentage = round(
            fake_count
            / total_articles
            * 100,
            2
        )

        real_percentage = round(
            real_count
            / total_articles
            * 100,
            2
        )

    else:

        fake_percentage = 0

        real_percentage = 0


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = None

    precision = None

    recall = None

    f1_score = None

    roc_auc = None

    cm_fake_fake = None

    cm_fake_real = None

    cm_real_fake = None

    cm_real_real = None


    if os.path.exists(
        METRICS_PATH
    ):

        metrics = joblib.load(
            METRICS_PATH
        )


        accuracy = metrics.get(
            "accuracy"
        )

        precision = metrics.get(
            "precision"
        )

        recall = metrics.get(
            "recall"
        )

        f1_score = metrics.get(
            "f1_score"
        )

        roc_auc = metrics.get(
            "roc_auc"
        )


        cm = metrics.get(
            "confusion_matrix"
        )


        if cm is not None:

            cm_fake_fake = cm[0][0]

            cm_fake_real = cm[0][1]

            cm_real_fake = cm[1][0]

            cm_real_real = cm[1][1]


    return render_template(

        "dashboard.html",

        total_articles=total_articles,

        fake_count=fake_count,

        real_count=real_count,

        fake_percentage=fake_percentage,

        real_percentage=real_percentage,

        accuracy=accuracy,

        precision=precision,

        recall=recall,

        f1_score=f1_score,

        roc_auc=roc_auc,

        cm_fake_fake=cm_fake_fake,

        cm_fake_real=cm_fake_real,

        cm_real_fake=cm_real_fake,

        cm_real_real=cm_real_real,

        model_name=(
            "Calibrated Linear SVM"
        ),

        feature_method="TF-IDF",

        preprocessing=(
            "Natural Language Processing"
        ),

        classification=(
            "Binary Classification"
        )

    )


# ============================================================
# MODEL COMPARISON DATA
# ============================================================

@app.route(
    "/model-comparison-data"
)
def model_comparison_data():

    print(
        "\nLoading model comparison data..."
    )


    # --------------------------------------------------------
    # Check file
    # --------------------------------------------------------

    if not os.path.exists(
        COMPARISON_PATH
    ):

        print(
            "Comparison file not found:",
            COMPARISON_PATH
        )


        return jsonify({
            "error":
            "model_comparison.csv not found"
        }), 404


    try:

        # ----------------------------------------------------
        # Load CSV
        # ----------------------------------------------------

        df = pd.read_csv(
            COMPARISON_PATH
        )


        print(
            "Comparison columns:",
            list(df.columns)
        )


        # ----------------------------------------------------
        # Required columns
        # ----------------------------------------------------

        required_columns = [

            "Model",

            "Accuracy",

            "Precision",

            "Recall",

            "F1_Score"

        ]


        for column in required_columns:

            if column not in df.columns:

                return jsonify({

                    "error":
                    f"Missing column: {column}"

                }), 500


        # ----------------------------------------------------
        # Create response
        # ----------------------------------------------------

        results = []


        for _, row in df.iterrows():

            results.append({

                "Model": str(
                    row["Model"]
                ),

                "Accuracy": round(
                    float(
                        row["Accuracy"]
                    ) * 100,
                    2
                ),

                "Precision": round(
                    float(
                        row["Precision"]
                    ) * 100,
                    2
                ),

                "Recall": round(
                    float(
                        row["Recall"]
                    ) * 100,
                    2
                ),

                "F1_Score": round(
                    float(
                        row["F1_Score"]
                    ) * 100,
                    2
                )

            })


        print(
            "Comparison data loaded successfully."
        )


        return jsonify(
            results
        )


    except Exception as error:

        print(
            "Model comparison error:",
            error
        )


        return jsonify({

            "error": str(error)

        }), 500


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("STARTING FLASK APPLICATION")
    print("=" * 60)

    print()

    print(
        "Detection:"
    )

    print(
        "http://127.0.0.1:5000/"
    )

    print()

    print(
        "Dashboard:"
    )

    print(
        "http://127.0.0.1:5000/dashboard"
    )

    print()

    print(
        "Model Comparison API:"
    )

    print(
        "http://127.0.0.1:5000/model-comparison-data"
    )

    print()

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True

    )