
import pandas as pd
import numpy as np

SCORES_PATH = "data/processed/multilabel_theme_scores.parquet"
REVIEWS_PATH = "data/processed/semantic_complaint_classifications.parquet"

OUTPUT_PATH = "data/processed/final_complaint_themes.parquet"
SUMMARY_PATH = "reports/final_complaint_theme_summary.csv"

# Provisional thresholds; not accuracy-calibrated.
THRESHOLDS = {
    "App_performance": 0.40,
    "Login_authentication": 0.40,
    "Account_restriction": 0.40,
    "Fraud_security": 0.40,
    "Money_transfer": 0.40,
    "Customer_support": 0.40,
    "Card_payment": 0.40,
    "Features_usability": 0.40,
    "Trading_investment": 0.40,
}

scores = pd.read_parquet(SCORES_PATH)
reviews = pd.read_parquet(REVIEWS_PATH)

df = reviews.merge(
    scores,
    on="review_id",
    how="inner",
    validate="one_to_one"
)

score_columns = {
    col.replace("score__", ""): col
    for col in df.columns
    if col.startswith("score__")
}

if set(score_columns) != set(THRESHOLDS):
    raise ValueError("Theme score columns do not match thresholds.")

for theme, column in score_columns.items():
    df[f"theme__{theme}"] = (
        df[column] >= THRESHOLDS[theme]
    )

theme_columns = [
    f"theme__{theme}" for theme in THRESHOLDS
]

df["assigned_theme_count"] = (
    df[theme_columns].sum(axis=1)
)

df["assigned_themes"] = df.apply(
    lambda row: [
        theme for theme in THRESHOLDS
        if row[f"theme__{theme}"]
    ],
    axis=1
)

df["theme_status"] = np.select(
    [
        df["assigned_theme_count"].eq(0),
        df["assigned_theme_count"].eq(1),
        df["assigned_theme_count"].gt(1),
    ],
    [
        "unassigned",
        "single_theme",
        "multiple_themes",
    ],
    default="unknown"
)

df.to_parquet(OUTPUT_PATH, index=False)

summary = pd.DataFrame([
    {
        "theme": theme,
        "threshold": THRESHOLDS[theme],
        "assigned_reviews": int(
            df[f"theme__{theme}"].sum()
        ),
        "coverage_percentage": round(
            df[f"theme__{theme}"].mean() * 100, 2
        ),
    }
    for theme in THRESHOLDS
])

summary.to_csv(SUMMARY_PATH, index=False)

print("FINAL MULTI-LABEL CLASSIFICATION")
print("=" * 55)
print("Total reviews:", len(df))
print("\nAssignment status:")
print(df["theme_status"].value_counts().to_string())

print("\nTheme summary:")
print(summary.to_string(index=False))

print("\nAverage themes per review:",
      round(df["assigned_theme_count"].mean(), 3))

print("\nSaved:", OUTPUT_PATH)
print("Saved:", SUMMARY_PATH)
