
from pathlib import Path
import regex
import unicodedata
import re
import pandas as pd
import emoji

PROCESSED_DIR = Path("data/processed")


def clean_text(text):
    if pd.isna(text):
        return ""

    # Normalize Unicode characters
    text = unicodedata.normalize("NFKC", str(text))

    # Convert to lowercase
    text = text.lower()

    # Convert emojis into readable words
    text = emoji.demojize(text, delimiters=(" ", " "))

    # Replace emoji underscores with spaces
    text = text.replace("_", " ")

    # Remove URLs
    text = regex.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
        flags=regex.IGNORECASE
    )

    # Remove HTML tags
    text = regex.sub(r"<[^>]+>", " ", text)

    # Preserve letters and numbers from ALL languages
    text = regex.sub(
    r"[^\p{L}\p{M}\p{N}\s$%.'+\-]",
    " ",
    text
)

    # Remove extra spaces
    text = regex.sub(r"\s+", " ", text)

    return text.strip()


def process_dataset(input_file, output_file, text_column, name):
    print(f"\nCleaning {name}...")

    df = pd.read_parquet(PROCESSED_DIR / input_file)

    # Preserve original text
    df["cleaned_text"] = df[text_column].apply(clean_text)

    # Track whether usable text exists after cleaning
    df["has_cleaned_text"] = df["cleaned_text"].ne("")


    df["has_cleaned_text"] = df["cleaned_text"].str.len() > 0

    print(f"Total records: {len(df):,}")
    print("Empty cleaned records:", (~df["has_cleaned_text"]).sum())

    print("\nOriginal vs cleaned examples:")

    for _, row in df.head(5).iterrows():
        print("\nOriginal:", row[text_column])
        print("Cleaned: ", row["cleaned_text"])

    # Show some emoji-only examples
    emoji_examples = df[
    df[text_column].apply(
        lambda value: bool(emoji.emoji_list(str(value)))
    )
    ].head(5)

    print("\nEmoji-containing examples:")

    for _, row in emoji_examples.iterrows():
        print("Original:", row[text_column])
        print("Cleaned: ", row["cleaned_text"])

    output_path = PROCESSED_DIR / output_file
    df.to_parquet(output_path, index=False)

    print(f"\nSaved: {output_path}")


def main():
    process_dataset(
        "banking_reviews_prepared.parquet",
        "banking_reviews_cleaned.parquet",
        "review_text",
        "banking reviews"
    )

    process_dataset(
        "financial_news_prepared.parquet",
        "financial_news_cleaned.parquet",
        "sentence",
        "financial news"
    )

    print("\nTEXT PREPROCESSING COMPLETED.")


if __name__ == "__main__":
    main()
