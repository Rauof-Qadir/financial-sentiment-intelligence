import pandas as pd

DATA_PATH = "data/processed/banking_complaints.parquet"

print("\nLoading complaint candidates...")

df = pd.read_parquet(DATA_PATH)

# 1. Rating distribution
print("\n1. Star rating distribution")

rating_counts = df["score"].value_counts().sort_index()

print(rating_counts.to_string())

print("\nRating percentages:")

print(
    df["score"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
    .round(2)
    .to_string()
)

# 2. Rating 4 and 5 candidates
print("\n2. Negative annotations with 4 or 5 stars")

high_rating = df[df["score"].isin([4, 5])]

print("Total:", len(high_rating))
print("Percentage:", round(len(high_rating) / len(df) * 100, 2), "%")

# 3. Short reviews
print("\n3. Short review candidates (< 3 words)")

short_reviews = df[df["word_count"] < 3]

print("Total:", len(short_reviews))

for _, row in short_reviews.sample(
    n=min(20, len(short_reviews)),
    random_state=42
).iterrows():

    print(
        f"\nRating: {row['score']} | "
        f"App: {row['app']} | "
        f"Review: {row['cleaned_text']}"
    )

# 4. High-rating examples
print("\n4. Random 4–5 star examples")

for _, row in high_rating.sample(
    n=min(30, len(high_rating)),
    random_state=42
).iterrows():

    print(
        f"\nRating: {row['score']} | "
        f"App: {row['app']} | "
        f"Review: {row['cleaned_text'][:250]}"
    )

print("\nCOMPLAINT CANDIDATE AUDIT COMPLETED.")