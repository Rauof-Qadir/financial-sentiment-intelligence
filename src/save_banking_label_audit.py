
import pandas as pd
import json
from pathlib import Path

DATA_PATH = "data/processed/banking_reviews_cleaned.parquet"
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("\nLoading banking reviews...")

df = pd.read_parquet(DATA_PATH)

rating_label = df["rating_sentiment"]
text_label = df["review_sentiment_label"]

agreement_table = pd.crosstab(
    rating_label,
    text_label
)

row_percentages = (
    agreement_table
    .div(agreement_table.sum(axis=1), axis=0)
    * 100
)

agreement = (rating_label == text_label).mean()

neutral = df[rating_label == "neutral"]

neutral_conflicts = neutral[
    neutral["rating_sentiment"]
    != neutral["review_sentiment_label"]
]

report = {
    "total_records": int(len(df)),
    "overall_agreement_percentage": round(
        float(agreement * 100), 2
    ),
    "rating_label_counts": {
        str(k): int(v)
        for k, v in rating_label.value_counts().items()
    },
    "existing_annotation_counts": {
        str(k): int(v)
        for k, v in text_label.value_counts().items()
    },
    "neutral_rating_records": int(len(neutral)),
    "neutral_rating_conflicts": int(len(neutral_conflicts)),
    "neutral_rating_agreement_percentage": round(
        float(row_percentages.loc["neutral", "neutral"]), 2
    ),
    "row_wise_agreement_percentages": (
        row_percentages.round(2).to_dict(orient="index")
    ),
    "interpretation": (
        "Rating-derived sentiment and existing text annotations "
        "measure related but different signals. Neither should be "
        "assumed to be perfect ground truth."
    )
}

output_path = REPORT_DIR / "banking_label_audit.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4, ensure_ascii=False)

agreement_table.to_csv(
    REPORT_DIR / "banking_label_agreement.csv"
)

print("\nAudit report saved:", output_path)
print("Agreement table saved.")
print("\nOverall agreement:", report["overall_agreement_percentage"], "%")
print("Neutral conflicts:", report["neutral_rating_conflicts"])

print("\nLABEL AUDIT REPORT COMPLETED.")
