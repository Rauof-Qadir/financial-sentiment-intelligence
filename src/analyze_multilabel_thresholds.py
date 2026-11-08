
import numpy as np
import pandas as pd

INPUT_PATH = "data/processed/multilabel_theme_scores.parquet"
OUTPUT_PATH = "reports/multilabel_threshold_analysis.csv"

df = pd.read_parquet(INPUT_PATH)

score_columns = [
    column for column in df.columns
    if column.startswith("score__")
]

thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

results = []

scores = df[score_columns].to_numpy()

for threshold in thresholds:
    predictions = scores >= threshold

    labels_per_review = predictions.sum(axis=1)

    results.append({
        "threshold": threshold,
        "reviews_with_no_theme": int(
            (labels_per_review == 0).sum()
        ),
        "reviews_with_one_theme": int(
            (labels_per_review == 1).sum()
        ),
        "reviews_with_multiple_themes": int(
            (labels_per_review > 1).sum()
        ),
        "average_themes_per_review": round(
            labels_per_review.mean(), 3
        ),
        "total_theme_assignments": int(
            predictions.sum()
        )
    })

result_df = pd.DataFrame(results)

result_df.to_csv(OUTPUT_PATH, index=False)

print("\nMULTI-LABEL THRESHOLD ANALYSIS")
print("=" * 90)
print(result_df.to_string(index=False))

print("\nSaved:", OUTPUT_PATH)
