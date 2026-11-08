
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT_DIR / "data" / "processed"
    / "banking_reviews_cleaned.parquet"
)

OUTPUT_PATH = (
    ROOT_DIR / "data" / "processed"
    / "banking_reviews_with_splits.parquet"
)

print("Loading banking reviews...")

df = pd.read_parquet(DATA_PATH)

df["review_date"] = pd.to_datetime(
    df["review_date"],
    errors="coerce"
)

df = df.dropna(
    subset=["review_date", "rating_sentiment"]
).copy()

# Sort oldest to newest
df = df.sort_values(
    "review_date"
).reset_index(drop=True)

# Calculate chronological boundaries
n = len(df)

train_position = int(n * 0.70)
validation_position = int(n * 0.85)

train_end_date = df.loc[
    train_position - 1, "review_date"
]

validation_end_date = df.loc[
    validation_position - 1, "review_date"
]

# Assign splits
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

# Save dataset with split column
df.to_parquet(
    OUTPUT_PATH,
    index=False
)

print("\nSaved:", OUTPUT_PATH)

print("\nSplit counts:")
print(df["split"].value_counts())

print("\nSplit date boundaries:")

for split in ["train", "validation", "test"]:
    subset = df[df["split"] == split]

    print(
        split,
        "|",
        subset["review_date"].min(),
        "to",
        subset["review_date"].max()
    )

print("\nBanking review splits saved successfully.")
