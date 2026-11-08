
import pandas as pd

INPUT_PATH = "data/processed/core_complaints_clustered_k10_v3.parquet"
OUTPUT_PATH = "reports/mixed_cluster_validation_sample.csv"

df = pd.read_parquet(INPUT_PATH)

# Only investigate the three mixed clusters.
mixed_clusters = [1, 2, 5]

samples = []

for cluster_id in mixed_clusters:

    cluster_df = df[df["cluster_id"] == cluster_id]

    sample = cluster_df.sample(
        n=min(30, len(cluster_df)),
        random_state=42
    ).copy()

    samples.append(sample)

validation_df = pd.concat(samples, ignore_index=True)

validation_df = validation_df[
    ["cluster_id", "app", "score", "cleaned_text"]
].copy()

# Columns for manual business validation.
validation_df["manual_theme"] = ""
validation_df["notes"] = ""

validation_df.to_csv(OUTPUT_PATH, index=False)

print("CLUSTER VALIDATION SAMPLE CREATED")
print("=" * 50)

print(f"Total sampled reviews: {len(validation_df)}")

print("\nSamples per cluster:")
print(validation_df["cluster_id"].value_counts().sort_index())

print("\nSaved:", OUTPUT_PATH)
