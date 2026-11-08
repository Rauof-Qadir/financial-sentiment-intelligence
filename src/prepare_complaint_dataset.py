import pandas as pd

DATA_PATH = "data/processed/banking_reviews_cleaned.parquet"
OUTPUT_PATH = "data/processed/banking_complaints.parquet"

print("\nLoading banking reviews...")

df = pd.read_parquet(DATA_PATH)

print("Original records:", len(df))

# Keep negative text annotations with usable text
complaints = df[
    (df["review_sentiment_label"] == "negative")
    & (df["has_cleaned_text"])
].copy()

# Remove empty text just in case
complaints = complaints[
    complaints["cleaned_text"].str.strip().ne("")
].copy()

# Calculate text length
complaints["word_count"] = (
    complaints["cleaned_text"].str.split().str.len()
)

# Save the complaint dataset
complaints.to_parquet(
    OUTPUT_PATH,
    index=False
)

print("\nCOMPLAINT DATASET SUMMARY")
print("-" * 50)

print("Total complaint candidates:", len(complaints))

print("\nReviews by banking app:")
print(complaints["app"].value_counts().to_string())

print("\nComplaint date range:")
print("Start:", complaints["review_date"].min())
print("End:", complaints["review_date"].max())

print("\nWord count statistics:")
print(complaints["word_count"].describe().round(2))

print("\nReviews with fewer than 3 words:")
print((complaints["word_count"] < 3).sum())

print("\nRandom complaint examples:")

samples = complaints.sample(
    n=min(15, len(complaints)),
    random_state=42
)

for _, row in samples.iterrows():

    print("\nApp:", row["app"])
    print("Rating:", row["score"])
    print("Words:", row["word_count"])
    print("Review:", row["cleaned_text"][:400])

print("\nCOMPLAINT DATASET PREPARATION COMPLETED.")