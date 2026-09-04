import pandas as pd
import os

print("=" * 60)
print("FAKE NEWS DATASET PREPARATION")
print("=" * 60)

# ------------------------------------------------------------
# File paths
# ------------------------------------------------------------

fake_path = "data/raw/Fake.csv"
real_path = "data/raw/True.csv"

output_path = "data/processed_news.csv"

# ------------------------------------------------------------
# Check files
# ------------------------------------------------------------

if not os.path.exists(fake_path):
    raise FileNotFoundError(
        f"Fake.csv not found: {fake_path}"
    )

if not os.path.exists(real_path):
    raise FileNotFoundError(
        f"True.csv not found: {real_path}"
    )

# ------------------------------------------------------------
# Load datasets
# ------------------------------------------------------------

print("\nLoading FAKE news dataset...")

fake = pd.read_csv(fake_path)

print("Loading REAL news dataset...")

real = pd.read_csv(real_path)

print("\nOriginal dataset sizes:")
print("FAKE:", len(fake))
print("REAL:", len(real))

# ------------------------------------------------------------
# Add labels explicitly
# ------------------------------------------------------------

fake["label"] = "FAKE"
real["label"] = "REAL"

# ------------------------------------------------------------
# Keep required columns
# ------------------------------------------------------------

required_columns = [
    "title",
    "text",
    "subject",
    "date",
    "label"
]

fake = fake[required_columns]

real = real[required_columns]

# ------------------------------------------------------------
# Combine datasets
# ------------------------------------------------------------

df = pd.concat(
    [fake, real],
    ignore_index=True
)

# ------------------------------------------------------------
# Remove missing values
# ------------------------------------------------------------

df["title"] = df["title"].fillna("")

df["text"] = df["text"].fillna("")

df["subject"] = df["subject"].fillna("")

df["date"] = df["date"].fillna("")

# ------------------------------------------------------------
# Create combined text
# ------------------------------------------------------------

df["combined_text"] = (
    df["title"].astype(str)
    + " "
    + df["text"].astype(str)
)

# ------------------------------------------------------------
# Remove empty articles
# ------------------------------------------------------------

df = df[
    df["combined_text"].str.strip().str.len() > 0
]

# ------------------------------------------------------------
# Shuffle dataset
# ------------------------------------------------------------

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# ------------------------------------------------------------
# Save dataset
# ------------------------------------------------------------

df.to_csv(
    output_path,
    index=False
)

# ------------------------------------------------------------
# Display results
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DATASET PREPARATION COMPLETED")
print("=" * 60)

print("\nFinal dataset shape:")
print(df.shape)

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nColumns:")
print(list(df.columns))

print("\nSaved to:")
print(output_path)

print("\nFirst 5 records:")
print(
    df[
        ["title", "label"]
    ].head().to_string(index=False)
)

print("\n" + "=" * 60)