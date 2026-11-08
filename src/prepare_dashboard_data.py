
import os
import pandas as pd

INPUT_PATH = "data/processed/final_complaint_themes.parquet"
OUTPUT_DIR = "data/processed/dashboard"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_parquet(INPUT_PATH)

theme_columns = [
    col for col in df.columns
    if col.startswith("theme__")
]

# Overall theme counts
theme_summary = pd.DataFrame({
    "theme": [
        col.replace("theme__", "").replace("_", " ")
        for col in theme_columns
    ],
    "reviews": [
        int(df[col].sum())
        for col in theme_columns
    ]
})

theme_summary["coverage_pct"] = (
    theme_summary["reviews"] / len(df) * 100
)

theme_summary = theme_summary.sort_values(
    "reviews", ascending=False
)

theme_summary.to_csv(
    f"{OUTPUT_DIR}/theme_summary.csv",
    index=False
)

# App-level complaint counts
app_summary = (
    df.groupby("app")
    .agg(
        total_reviews=("review_id", "count"),
        average_rating=("score", "mean")
    )
    .reset_index()
)

app_summary.to_csv(
    f"{OUTPUT_DIR}/app_summary.csv",
    index=False
)

# App x theme matrix
app_theme = (
    df.groupby("app")[theme_columns]
    .sum()
    .reset_index()
)

app_theme.to_csv(
    f"{OUTPUT_DIR}/app_theme_summary.csv",
    index=False
)

# Review-level dashboard data
review_columns = [
    col for col in [
        "review_id",
        "app",
        "score",
        "review_date",
        "review_text",
        "assigned_themes",
        "assigned_theme_count",
        "theme_status"
    ]
    if col in df.columns
]

reviews = df[review_columns].copy()

reviews.to_parquet(
    f"{OUTPUT_DIR}/dashboard_reviews.parquet",
    index=False
)

print("DASHBOARD DATA PREPARATION")
print("=" * 50)
print("Total reviews:", len(df))
print("Apps:", df["app"].nunique())
print("Themes:", len(theme_columns))

print("\nTheme summary:")
print(theme_summary.to_string(index=False))

print("\nApp summary:")
print(app_summary.to_string(index=False))

print("\nSaved dashboard files in:", OUTPUT_DIR)
