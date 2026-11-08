
from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def prepare_banking_reviews():
    print("\nPreparing banking reviews...")

    df = pd.read_parquet(
        RAW_DIR / "banking_app_reviews.parquet"
    )

    original_rows = len(df)

    # Keep records containing usable review text
    df["review_text"] = df["review_text"].fillna("").str.strip()
    df = df[df["review_text"] != ""].copy()

    # Derive satisfaction categories from explicit star ratings
    df["rating_sentiment"] = pd.NA

    df.loc[df["score"].isin([1, 2]), "rating_sentiment"] = "negative"
    df.loc[df["score"] == 3, "rating_sentiment"] = "neutral"
    df.loc[df["score"].isin([4, 5]), "rating_sentiment"] = "positive"

    # Normalize text fields
    df["app"] = df["app"].str.strip()

    # Flag invalid reply times; do not modify original values
    df["invalid_reply_time"] = df["time_to_reply(h)"] < 0

    # Select useful columns and preserve original annotations
    selected_columns = [
        "review_id",
        "app",
        "score",
        "review_text",
        "review_date",
        "thumbs_up",
        "replied",
        "reply_text",
        "reply_date",
        "time_to_reply(h)",
        "invalid_reply_time",
        "review_sentiment",
        "review_sentiment_label",
        "rating_sentiment",
        "topic_label_ALL",
    ]

    prepared = df[selected_columns].copy()

    output_path = PROCESSED_DIR / "banking_reviews_prepared.parquet"
    prepared.to_parquet(output_path, index=False)

    print(f"Original rows: {original_rows:,}")
    print(f"Rows with usable text: {len(prepared):,}")
    print(f"Rows removed due to empty text: {original_rows - len(prepared):,}")

    print("\nRating-derived satisfaction distribution:")
    print(prepared["rating_sentiment"].value_counts())

    print("\nExisting sentiment annotation distribution:")
    print(prepared["review_sentiment_label"].value_counts())

    print("\nInvalid reply times:")
    print(prepared["invalid_reply_time"].sum())

    print(f"\nSaved: {output_path}")


def prepare_financial_news():
    print("\nPreparing financial news...")

    splits = {
        "train": "financial_news_train.parquet",
        "validation": "financial_news_validation.parquet",
        "test": "financial_news_test.parquet",
    }

    prepared_splits = []

    for split_name, filename in splits.items():
        df = pd.read_parquet(RAW_DIR / filename)

        # Normalize sentence whitespace
        df["sentence"] = df["sentence"].str.strip()

        # Preserve the original split
        df["split"] = split_name

        prepared_splits.append(df)

    prepared = pd.concat(prepared_splits, ignore_index=True)

    # Verify the known label mapping
    expected_mapping = {
        0: "negative",
        1: "neutral",
        2: "positive",
    }

    for label, sentiment in expected_mapping.items():
        actual = prepared.loc[
            prepared["label"] == label, "sentiment"
        ].unique()

        if len(actual) != 1 or actual[0] != sentiment:
            raise ValueError(
                f"Unexpected label mapping for {label}: {actual}"
            )

    output_path = PROCESSED_DIR / "financial_news_prepared.parquet"
    prepared.to_parquet(output_path, index=False)

    print(f"Total news records: {len(prepared):,}")

    print("\nRecords by split:")
    print(prepared["split"].value_counts())

    print("\nSentiment distribution:")
    print(prepared["sentiment"].value_counts())

    print(f"\nSaved: {output_path}")


def main():
    prepare_banking_reviews()
    prepare_financial_news()

    print("\nDATA PREPARATION COMPLETED.")


if __name__ == "__main__":
    main()
