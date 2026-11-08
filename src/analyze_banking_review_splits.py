
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT_DIR / "data" / "processed"
    / "banking_reviews_cleaned.parquet"
)

df = pd.read_parquet(DATA_PATH)

# 1. Prepare dates
df["review_date"] = pd.to_datetime(
    df["review_date"],
    errors="coerce"
)

print("\nDATA QUALITY CHECK")

print("Total records:", len(df))
print("Missing dates:", df["review_date"].isna().sum())
print("Missing labels:", df["rating_sentiment"].isna().sum())
print("Duplicate review IDs:", df["review_id"].duplicated().sum())
print("Empty cleaned text:", (~df["has_cleaned_text"]).sum())

# 2. Keep records with valid date and label
df = df.dropna(
    subset=["review_date", "rating_sentiment"]
).copy()

# 3. Sort chronologically
df = df.sort_values(
    "review_date",
    ascending=True
).reset_index(drop=True)

# 4. Calculate date boundaries
n = len(df)

train_position = int(n * 0.70)
validation_position = int(n * 0.85)

train_end_date = df.loc[
    train_position - 1, "review_date"
]

validation_end_date = df.loc[
    validation_position - 1, "review_date"
]

# 5. Assign chronological splits
df["split"] = "test"

df.loc[
    df["review_date"] <= train_end_date,
    "split"
] = "train"

df.loc[
    (df["review_date"] > train_end_date)
    & (df["review_date"] <= validation_end_date),
    "split"
] = "validation"

# 6. Print split sizes and date ranges
print("\nCHRONOLOGICAL SPLIT SUMMARY")

for split_name in ["train", "validation", "test"]:

    subset = df[df["split"] == split_name]

    print(f"\n{split_name.upper()}")

    print("Records:", len(subset))

    print(
        "Date range:",
        subset["review_date"].min(),
        "to",
        subset["review_date"].max()
    )

    print("\nSentiment distribution:")

    print(
        subset["rating_sentiment"]
        .value_counts()
        .to_string()
    )

    print("\nSentiment percentages:")

    print(
        (
            subset["rating_sentiment"]
            .value_counts(normalize=True) * 100
        ).round(2).to_string()
    )

# 7. Check app distribution across splits
print("\nAPP DISTRIBUTION BY SPLIT")

print(
    pd.crosstab(
        df["split"],
        df["app"],
        normalize="index"
    ).mul(100).round(2)
)

# 8. Verify chronological integrity
print("\nCHRONOLOGICAL INTEGRITY")

print("Train maximum:", df[df["split"] == "train"]["review_date"].max())
print("Validation minimum:", df[df["split"] == "validation"]["review_date"].min())
print("Validation maximum:", df[df["split"] == "validation"]["review_date"].max())
print("Test minimum:", df[df["split"] == "test"]["review_date"].min())

print("\nSPLIT ANALYSIS COMPLETED")
