import pandas as pd
import matplotlib.pyplot as plt
import os


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "model_comparison.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "static",
    "model_comparison.png"
)


# ============================================================
# CHECK INPUT FILE
# ============================================================

if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"Model comparison file not found:\n{INPUT_FILE}"
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("MODEL COMPARISON GRAPH")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)


print("\nLoaded results:")
print(
    df.to_string(index=False)
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score"
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:

    raise ValueError(
        "Missing columns: "
        + ", ".join(missing)
    )


# ============================================================
# CONVERT VALUES TO PERCENTAGES
# ============================================================

models = df["Model"].astype(str)

accuracy = df["Accuracy"] * 100

precision = df["Precision"] * 100

recall = df["Recall"] * 100

f1_score = df["F1_Score"] * 100


# ============================================================
# CREATE GRAPH
# ============================================================

plt.figure(
    figsize=(11, 6)
)


x = list(
    range(len(models))
)


width = 0.18


plt.bar(
    [i - 1.5 * width for i in x],
    accuracy,
    width,
    label="Accuracy"
)


plt.bar(
    [i - 0.5 * width for i in x],
    precision,
    width,
    label="Precision"
)


plt.bar(
    [i + 0.5 * width for i in x],
    recall,
    width,
    label="Recall"
)


plt.bar(
    [i + 1.5 * width for i in x],
    f1_score,
    width,
    label="F1 Score"
)


# ============================================================
# LABELS
# ============================================================

plt.title(
    "Machine Learning Model Comparison"
)

plt.xlabel(
    "Machine Learning Model"
)

plt.ylabel(
    "Performance (%)"
)


plt.xticks(
    x,
    models,
    rotation=15
)


plt.ylim(
    0,
    100
)


plt.legend()

plt.grid(
    axis="y",
    alpha=0.2
)


plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

plt.savefig(
    OUTPUT_FILE,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# COMPLETED
# ============================================================

print()
print("=" * 60)

print(
    "GRAPH CREATED SUCCESSFULLY"
)

print("=" * 60)

print()

print(
    "Saved to:"
)

print(
    OUTPUT_FILE
)

print()