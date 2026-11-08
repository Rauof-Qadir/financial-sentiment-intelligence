
import pandas as pd

INPUT_PATH = "data/processed/theme_validation_sample.csv"
OUTPUT_PATH = "reports/theme_validation_worksheet.csv"

df = pd.read_csv(INPUT_PATH)

worksheet = df[
    [
        "review_id",
        "app",
        "score",
        "candidate_theme",
        "score_band",
        "review_text",
    ]
].copy()

worksheet["similarity_score"] = None

# Recover candidate theme's actual similarity score.
scores = pd.read_parquet(
    "data/processed/multilabel_theme_scores.parquet"
)

scores = scores.set_index("review_id")

for index, row in worksheet.iterrows():
    column = "score__" + row["candidate_theme"]

    worksheet.at[index, "similarity_score"] = scores.loc[
        row["review_id"], column
    ]

worksheet["human_relevant"] = ""
worksheet["human_correct_theme"] = ""
worksheet["notes"] = ""

worksheet.to_csv(OUTPUT_PATH, index=False)

print("Worksheet rows:", len(worksheet))
print("Unique reviews:", worksheet["review_id"].nunique())
print("Saved:", OUTPUT_PATH)
print("\nColumns:")
print(worksheet.columns.tolist())
