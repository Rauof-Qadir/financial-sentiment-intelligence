
import pandas as pd

WORKSHEET_PATH = "reports/theme_validation_worksheet.csv"
REVIEWS_PATH = "data/processed/semantic_complaint_classifications.parquet"
OUTPUT_PATH = "reports/theme_validation_with_weak_evidence.csv"

worksheet = pd.read_csv(WORKSHEET_PATH)

reviews = pd.read_parquet(REVIEWS_PATH)

required = ["review_id", "topic_label_ALL"]

missing = [
    col for col in required
    if col not in reviews.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")

evidence = reviews[required].drop_duplicates("review_id")

result = worksheet.merge(
    evidence,
    on="review_id",
    how="left",
    validate="many_to_one"
)

result["weak_topic_match"] = (
    result["candidate_theme"]
    .str.replace("_", " ", regex=False)
    .str.lower()
    .str.strip()
    ==
    result["topic_label_ALL"]
    .fillna("")
    .str.lower()
    .str.strip()
)

result.to_csv(OUTPUT_PATH, index=False)

print("Rows:", len(result))
print("Unique reviews:", result["review_id"].nunique())
print("\nWeak topic evidence:")
print(result["weak_topic_match"].value_counts(dropna=False))

print("\nTopic label distribution:")
print(result["topic_label_ALL"].value_counts(dropna=False).head(15))

print("\nSaved:", OUTPUT_PATH)
