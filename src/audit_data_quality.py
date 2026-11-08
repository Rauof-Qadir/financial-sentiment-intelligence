
from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw")


def audit_banking_reviews():
    print("\n" + "=" * 65)
    print("BANKING REVIEWS: DETAILED QUALITY AUDIT")
    print("=" * 65)

    df = pd.read_parquet(DATA_DIR / "banking_app_reviews.parquet")

    # 1. Customer rating distribution
    print("\n1. STAR RATING DISTRIBUTION")
    print(df["score"].value_counts().sort_index())

    # 2. Existing sentiment labels
    print("\n2. REVIEW SENTIMENT LABEL DISTRIBUTION")
    print(df["review_sentiment_label"].value_counts(dropna=False))

    # 3. Compare rating with sentiment label
    print("\n3. STAR RATING vs SENTIMENT LABEL")
    print(pd.crosstab(
        df["score"],
        df["review_sentiment_label"],
        margins=True
    ))

    # 4. Missing review text
    print("\n4. MISSING REVIEW TEXT")
    print("Missing:", df["review_text"].isna().sum())
    print("Empty strings:", df["review_text"].fillna("").str.strip().eq("").sum())

    # 5. Reply status consistency
    print("\n5. REPLY STATUS CHECK")
    print("Replied flag counts:")
    print(df["replied"].value_counts(dropna=False))

    print("\nReply text missing by replied flag:")
    print(df.groupby("replied")["reply_text"].apply(
        lambda x: x.isna().sum()
    ))

    # 6. Investigate reply time
    print("\n6. REPLY TIME INVESTIGATION")
    print("Negative stored reply times:",
          (df["time_to_reply(h)"] < 0).sum())

    negative = df.loc[
        df["time_to_reply(h)"] < 0,
        ["review_date", "reply_date", "time_to_reply(h)"]
    ].head(10)

    print("\nFirst 10 negative reply-time records:")
    print(negative.to_string(index=False))

    # Compare stored reply time with timestamps
    valid_dates = df["reply_date"].notna() & df["review_date"].notna()

    actual_hours = (
        df.loc[valid_dates, "reply_date"]
        - df.loc[valid_dates, "review_date"]
    ).dt.total_seconds() / 3600

    print("\nActual timestamp difference (hours):")
    print(actual_hours.describe())

    print("\nStored reply time (hours):")
    print(df["time_to_reply(h)"].describe())


def audit_financial_news():
    print("\n" + "=" * 65)
    print("FINANCIAL NEWS: LABEL AND SPLIT AUDIT")
    print("=" * 65)

    filenames = {
        "train": "financial_news_train.parquet",
        "validation": "financial_news_validation.parquet",
        "test": "financial_news_test.parquet",
    }

    datasets = {}

    for split, filename in filenames.items():
        df = pd.read_parquet(DATA_DIR / filename)
        datasets[split] = df

        print(f"\n{split.upper()} CLASS DISTRIBUTION")
        print(df["sentiment"].value_counts())
        print("\nNumeric labels:")
        print(df["label"].value_counts().sort_index())

    # Check overlap between splits
    print("\nNEWS SENTENCE OVERLAP CHECK")

    train = set(datasets["train"]["sentence"].str.strip())
    validation = set(datasets["validation"]["sentence"].str.strip())
    test = set(datasets["test"]["sentence"].str.strip())

    print("Train-validation overlap:", len(train & validation))
    print("Train-test overlap:", len(train & test))
    print("Validation-test overlap:", len(validation & test))


def main():
    audit_banking_reviews()
    audit_financial_news()


if __name__ == "__main__":
    main()
