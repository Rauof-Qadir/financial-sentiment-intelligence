
import pandas as pd

DATA_PATH = "data/processed/banking_reviews_cleaned.parquet"

print("\nLoading banking reviews...")

df = pd.read_parquet(DATA_PATH)

print("\nComparing rating-derived labels with existing labels...")

comparison = df[
    df["rating_sentiment"].notna()
    & df["review_sentiment_label"].notna()
].copy()

comparison = comparison[
    comparison["review_sentiment_label"] != "undefined"
]

print("\nOverall label counts:")
print("\nRating-derived:")
print(comparison["rating_sentiment"].value_counts())

print("\nExisting text annotation:")
print(comparison["review_sentiment_label"].value_counts())

print("\nConfusion / agreement table:")
cross_tab = pd.crosstab(
    comparison["rating_sentiment"],
    comparison["review_sentiment_label"],
    margins=True
)

print(cross_tab)

print("\nRow-wise percentages:")
print(
    pd.crosstab(
        comparison["rating_sentiment"],
        comparison["review_sentiment_label"],
        normalize="index"
    ).round(3) * 100
)

print("\nOverall agreement:")
agreement = (
    comparison["rating_sentiment"]
    == comparison["review_sentiment_label"]
).mean()

print(f"Agreement: {agreement:.2%}")

print("\nNeutral rating examples with conflicting annotations:")

conflicts = comparison[
    (comparison["rating_sentiment"] == "neutral")
    & (comparison["review_sentiment_label"] != "neutral")
]

print("Total conflicts:", len(conflicts))

for _, row in conflicts.head(20).iterrows():
    print("\nRating-derived:", row["rating_sentiment"])
    print("Existing annotation:", row["review_sentiment_label"])
    print("Review:", row["review_text"][:400])
    print("-" * 70)

print("\nLABEL AUDIT COMPLETED.")
