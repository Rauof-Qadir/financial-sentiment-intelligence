import pandas as pd

DATA_PATH = "data/processed/banking_reviews_cleaned.parquet"

print("\nLoading banking reviews...")

df = pd.read_parquet(DATA_PATH)

# Keep records with actual text
df = df[df["has_cleaned_text"]].copy()

print("\nUsable records:", len(df))

# --------------------------------------------------
# 1. Topic distribution
# --------------------------------------------------

print("\n1. Topic distribution (%)")

topic_distribution = (
    df["topic_label_ALL"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print(topic_distribution.to_string())

# --------------------------------------------------
# 2. Topic vs existing text annotations
# --------------------------------------------------

print("\n2. Topic vs existing text sentiment")

topic_sentiment = pd.crosstab(
    df["topic_label_ALL"],
    df["review_sentiment_label"],
    normalize="index"
).mul(100).round(2)

print(topic_sentiment.to_string())

# --------------------------------------------------
# 3. Topic vs star ratings
# --------------------------------------------------

print("\n3. Topic vs star ratings")

topic_ratings = pd.crosstab(
    df["topic_label_ALL"],
    df["score"],
    normalize="index"
).mul(100).round(2)

print(topic_ratings.to_string())

# --------------------------------------------------
# 4. Negative text annotations by topic
# --------------------------------------------------

negative_reviews = df[
    df["review_sentiment_label"] == "negative"
].copy()

print("\n4. Negative text annotations by topic")

negative_topic_counts = (
    negative_reviews["topic_label_ALL"]
    .value_counts()
)

print(negative_topic_counts.to_string())

# --------------------------------------------------
# 5. Random review samples from each topic
# --------------------------------------------------

print("\n5. Random review samples")

topics = df["topic_label_ALL"].value_counts().index

for topic in topics:

    print("\n" + "=" * 75)
    print("TOPIC:", topic)

    samples = df[
        df["topic_label_ALL"] == topic
    ].sample(
        n=min(5, (df["topic_label_ALL"] == topic).sum()),
        random_state=42
    )

    for _, row in samples.iterrows():

        print("\nApp:", row["app"])
        print("Rating:", row["score"])
        print("Text annotation:", row["review_sentiment_label"])
        print("Review:", row["review_text"][:300])

print("\nTOPIC QUALITY ANALYSIS COMPLETED.")