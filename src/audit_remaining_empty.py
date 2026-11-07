
import pandas as pd
from pathlib import Path

path = Path("data/processed/banking_reviews_cleaned.parquet")

df = pd.read_parquet(path)

empty = df[df["cleaned_text"].eq("")].copy()

print("=" * 60)
print("REMAINING EMPTY TEXT INVESTIGATION")
print("=" * 60)

print("Remaining records:", len(empty))

print("\nOriginal text examples:")

for i, value in enumerate(empty["review_text"].head(30), 1):
    print(f"{i}. {repr(value)}")

print("\nOriginal text length:")
print(empty["review_text"].str.len().describe())

print("\nStar rating distribution:")
print(empty["score"].value_counts().sort_index())

print("\nExisting sentiment labels:")
print(empty["review_sentiment_label"].value_counts())

print("\nInvestigation completed.")
