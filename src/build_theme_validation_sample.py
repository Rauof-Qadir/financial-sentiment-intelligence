
import os
import numpy as np
import pandas as pd

INPUT_PATH = "data/processed/multilabel_theme_scores.parquet"
REVIEW_PATH = "data/processed/semantic_complaint_classifications.parquet"

OUTPUT_PATH = "data/processed/theme_validation_sample.csv"

SAMPLE_PER_THEME = 40
RANDOM_STATE = 42

scores = pd.read_parquet(INPUT_PATH)
reviews = pd.read_parquet(REVIEW_PATH)

df = scores.merge(
    reviews[
        ["review_id", "app", "score", "review_text", "cleaned_text"]
    ],
    on="review_id",
    how="inner",
    validate="one_to_one"
)

score_columns = [
    col for col in df.columns
    if col.startswith("score__")
]

samples = []

for column in score_columns:
    theme = column.replace("score__", "")

    # Create score bands to sample easy and difficult cases.
    df["_score_band"] = pd.cut(
        df[column],
        bins=[-np.inf, 0.20, 0.30, 0.40, 0.50, np.inf],
        labels=["very_low", "low", "medium", "high", "very_high"]
    )

    for band in df["_score_band"].dropna().unique():
        group = df[df["_score_band"] == band]

        sample = group.sample(
            n=min(8, len(group)),
            random_state=RANDOM_STATE
        ).copy()

        sample["candidate_theme"] = theme
        sample["score_band"] = band

        samples.append(sample)

result = pd.concat(samples, ignore_index=True)

# Remove duplicate reviews selected for multiple themes.
result = result.drop_duplicates(
    subset=["review_id", "candidate_theme"]
)

# Add empty columns for independent human validation.
result["human_relevant"] = ""
result["human_theme_notes"] = ""

result.to_csv(OUTPUT_PATH, index=False)

print("Validation examples:", len(result))
print("Unique reviews:", result["review_id"].nunique())
print("Candidate themes:", result["candidate_theme"].nunique())
print("Saved:", OUTPUT_PATH)
print("\nSample distribution:")
print(
    result.groupby(
        ["candidate_theme", "score_band"],
        observed=True
    ).size().to_string()
)
