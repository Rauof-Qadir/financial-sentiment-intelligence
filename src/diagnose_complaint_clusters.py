
import pandas as pd

DATA_PATH = "data/processed/core_complaints_clustered_k12_v2.parquet"

print("Loading clustered complaints...")
df = pd.read_parquet(DATA_PATH)

df["word_count"] = df["cleaned_text"].str.split().str.len()

print("\nDATASET OVERVIEW")
print("=" * 50)

print(f"Total reviews: {len(df):,}")
print(f"Unique complaint texts: {df['cleaned_text'].nunique():,}")
print(
    f"Duplicate text records: "
    f"{df['cleaned_text'].duplicated().sum():,}"
)

print("\nREVIEW LENGTH DISTRIBUTION")
print("=" * 50)

print(df["word_count"].describe().round(2))

print("\nReviews by word count:")
print(
    pd.cut(
        df["word_count"],
        bins=[0, 3, 5, 10, 20, 50, 100, float("inf")],
        labels=["1-3", "4-5", "6-10", "11-20",
                "21-50", "51-100", "100+"]
    ).value_counts().sort_index()
)

print("\nCLUSTER DISTRIBUTION")
print("=" * 50)

cluster_summary = (
    df.groupby("cluster_id")
    .agg(
        reviews=("cleaned_text", "size"),
        unique_texts=("cleaned_text", "nunique"),
        avg_words=("word_count", "mean"),
        median_words=("word_count", "median")
    )
    .sort_values("reviews", ascending=False)
)

cluster_summary["percentage"] = (
    cluster_summary["reviews"] / len(df) * 100
)

print(cluster_summary.round(2).to_string())

print("\nRANDOM SAMPLES FROM LARGE CLUSTER")
print("=" * 50)

largest_cluster = cluster_summary.index[0]

samples = df[
    df["cluster_id"] == largest_cluster
].sample(
    n=min(15, (df["cluster_id"] == largest_cluster).sum()),
    random_state=42
)

for _, row in samples.iterrows():
    print(f"\nWords: {row['word_count']}")
    print(f"Rating: {row['score']}")
    print(f"Review: {row['cleaned_text'][:350]}")

print("\nDIAGNOSTIC COMPLETED.")
