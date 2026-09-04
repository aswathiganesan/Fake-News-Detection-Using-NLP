import pandas as pd
import joblib


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    "models/best_fake_news_model.pkl"
)


# ============================================================
# LOAD REAL NEWS
# ============================================================

real_df = pd.read_csv(
    "data/raw/True.csv"
)


# ============================================================
# LOAD FAKE NEWS
# ============================================================

fake_df = pd.read_csv(
    "data/raw/Fake.csv"
)


# ============================================================
# TEST REAL NEWS
# ============================================================

print("=" * 60)
print("REAL NEWS TEST")
print("=" * 60)

real_correct = 0

for i in range(10):

    text = str(
        real_df.iloc[i]["title"]
    ) + " " + str(
        real_df.iloc[i]["text"]
    )

    prediction = model.predict(
        [text]
    )[0]

    print()
    print("Article:", i + 1)
    print("Title:", real_df.iloc[i]["title"])
    print("Expected: REAL")
    print("Predicted:", prediction)

    if prediction == "REAL":
        real_correct += 1


# ============================================================
# TEST FAKE NEWS
# ============================================================

print()
print("=" * 60)
print("FAKE NEWS TEST")
print("=" * 60)

fake_correct = 0

for i in range(10):

    text = str(
        fake_df.iloc[i]["title"]
    ) + " " + str(
        fake_df.iloc[i]["text"]
    )

    prediction = model.predict(
        [text]
    )[0]

    print()
    print("Article:", i + 1)
    print("Title:", fake_df.iloc[i]["title"])
    print("Expected: FAKE")
    print("Predicted:", prediction)

    if prediction == "FAKE":
        fake_correct += 1


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("TEST SUMMARY")
print("=" * 60)

print(
    "REAL correctly detected:",
    real_correct,
    "/ 10"
)

print(
    "FAKE correctly detected:",
    fake_correct,
    "/ 10"
)

print(
    "Total correct:",
    real_correct + fake_correct,
    "/ 20"
)

print("=" * 60)