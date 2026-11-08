
import pandas as pd

DATA_PATH = "data/processed/banking_reviews_cleaned.parquet"

print("\nLoading banking reviews...")

df = pd.read_parquet(DATA_PATH)

print("\nDataset shape:", df.shape)

print("\nTopic column information:")

print("Column exists:", "topic_label_ALL" in df.columns)

if "topic_label_ALL" in df.columns:

    print("\nMissing topic labels:")
    print(df["topic_label_ALL"].isna().sum())

    print("\nUnique topic count:")
    print(df["topic_label_ALL"].nunique())

    print("\nTop 30 topics:")

    print(
        df["topic_label_ALL"]
        .value_counts()
        .head(30)
        .to_string()
    )

    print("\nSample reviews by topic:")

    topics = (
        df["topic_label_ALL"]
        .value_counts()
        .head(10)
        .index
    )

    for topic in topics:

        print(f"\n{'=' * 60}")
        print("TOPIC:", topic)

        samples = df[
            df["topic_label_ALL"] == topic
        ].head(3)

        for _, row in samples.iterrows():

            print("\nApp:", row["app"])
            print("Rating:", row["score"])
            print("Existing sentiment:", row["review_sentiment_label"])
            print("Review:", row["review_text"][:350])

print("\nTOPIC AUDIT COMPLETED.")
