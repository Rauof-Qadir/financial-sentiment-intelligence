
import pandas as pd

INPUT_PATH = "data/processed/multilabel_theme_scores.parquet"
OUTPUT_PATH = "reports/theme_specific_threshold_analysis.csv"

df = pd.read_parquet(INPUT_PATH)

thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

score_columns = [
    col for col in df.columns
    if col.startswith("score__")
]

results = []

for column in score_columns:
    theme = column.replace("score__", "").replace("_", " ")

    scores = df[column]

    for threshold in thresholds:
        matched = scores >= threshold

        results.append({
            "theme": theme,
            "threshold": threshold,
            "matched_reviews": int(matched.sum()),
            "coverage_percentage": round(
                matched.mean() * 100, 2
            ),
            "average_similarity_of_matched": round(
                scores[matched].mean(), 4
            ) if matched.any() else 0
        })

result = pd.DataFrame(results)

result.to_csv(OUTPUT_PATH, index=False)

print("\nTHEME-SPECIFIC THRESHOLD ANALYSIS")
print("=" * 90)

for theme, group in result.groupby("theme"):
    print(f"\n{theme}")
    print(group[
        [
            "threshold",
            "matched_reviews",
            "coverage_percentage",
            "average_similarity_of_matched"
        ]
    ].to_string(index=False))

print("\nSaved:", OUTPUT_PATH)
