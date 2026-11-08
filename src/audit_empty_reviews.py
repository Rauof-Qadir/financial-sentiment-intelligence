
import pandas as pd

FILE_PATH = "data/processed/banking_reviews_cleaned.parquet"

df = pd.read_parquet(FILE_PATH)

# Find reviews that became empty after preprocessing
empty_reviews = df[df["cleaned_text"].str.strip() == ""].copy()

print("=" * 60)
print("EMPTY CLEANED REVIEW INVESTIGATION")
print("=" * 60)

print(f"\nTotal empty cleaned reviews: {len(empty_reviews):,}")

print("\nOriginal text examples:")
for index, text in enumerate(empty_reviews["review_text"].head(30), start=1):
    print(f"{index}. {repr(text)}")

print("\nOriginal text length statistics:")
print(empty_reviews["review_text"].str.len().describe())

print("\nStar rating distribution:")
print(empty_reviews["score"].value_counts().sort_index())

print("\nExisting sentiment distribution:")
print(empty_reviews["review_sentiment_label"].value_counts())

print("\nApps containing empty cleaned reviews:")
print(empty_reviews["app"].value_counts())

print("\nInvestigation completed.")
