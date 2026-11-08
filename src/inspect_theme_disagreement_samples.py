
import pandas as pd

INPUT_PATH = "reports/theme_disagreement_samples.csv"
OUTPUT_PATH = "reports/top_theme_disagreement_examples.csv"

df = pd.read_csv(INPUT_PATH)

top_pairs = [
    ("Other/unclear", "App performance"),
    ("App performance", "Other/unclear"),
    ("Login/authentication", "App performance"),
]

selected = []

for keyword_theme, semantic_theme in top_pairs:
    group = df[
        (df["keyword_theme"] == keyword_theme)
        & (df["semantic_primary_theme"] == semantic_theme)
    ].copy()

    # Prefer reviews with a larger similarity margin.
    group = group.sort_values(
        ["low_confidence", "similarity_margin"],
        ascending=[True, False]
    )

    selected.append(group.head(10))

result = pd.concat(selected, ignore_index=True)

columns = [
    "review_id",
    "app",
    "score",
    "keyword_theme",
    "semantic_primary_theme",
    "semantic_secondary_theme",
    "semantic_similarity",
    "similarity_margin",
    "low_confidence",
    "review_text",
]

result = result[columns]
result.to_csv(OUTPUT_PATH, index=False)

print("Selected examples:", len(result))
print("Saved:", OUTPUT_PATH)

print("\nEXAMPLES")
print("=" * 80)

for _, row in result.iterrows():
    print("\nReview ID:", row["review_id"])
    print("Keyword:", row["keyword_theme"])
    print("Semantic:", row["semantic_primary_theme"])
    print("Similarity:", round(row["semantic_similarity"], 3))
    print("Margin:", round(row["similarity_margin"], 3))
    print("Review:", str(row["review_text"])[:350])
    print("-" * 80)
