
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans

DATA_PATH = "data/processed/banking_complaints_core.parquet"

OUTPUT_PATH = "data/processed/core_complaints_clustered_k10_v3.parquet"
REPORT_PATH = "reports/complaint_cluster_summary_k10_v3.csv"

N_CLUSTERS = 10
N_COMPONENTS = 100
RANDOM_STATE = 42

CUSTOM_STOPWORDS = {
    "bad", "terrible", "useless", "worst", "poor", "awful",
    "horrible", "disgusting", "pathetic", "excellent", "good",
    "great", "amazing", "brilliant", "app", "apps", "bank",
    "banking", "really", "absolutely", "totally", "very",
    "just", "thing", "things"
}

print("Loading core complaints...")

df = pd.read_parquet(DATA_PATH)

df = df[
    df["cleaned_text"].notna()
    & df["cleaned_text"].str.strip().ne("")
].copy()

df = df.reset_index(drop=True)

texts = df["cleaned_text"].astype(str)

print(f"Total complaints: {len(df):,}")

# --------------------------------------------------
# 1. TF-IDF FEATURES
# --------------------------------------------------

print("\nCreating TF-IDF features...")

stopwords = list(ENGLISH_STOP_WORDS.union(CUSTOM_STOPWORDS))

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.90,
    max_features=40000,
    stop_words=stopwords,
    sublinear_tf=True,
    dtype=np.float32
)

X = vectorizer.fit_transform(texts)

print(f"TF-IDF matrix: {X.shape}")

# Exclude rows that have no remaining features.
informative_mask = np.asarray(
    X.getnnz(axis=1)
).ravel() > 0

X_informative = X[informative_mask]

print(f"Informative reviews: {X_informative.shape[0]:,}")
print(f"Excluded zero-feature reviews: {(~informative_mask).sum():,}")

# --------------------------------------------------
# 2. DIMENSIONALITY REDUCTION
# --------------------------------------------------

print("\nApplying TruncatedSVD...")

svd = TruncatedSVD(
    n_components=N_COMPONENTS,
    n_iter=7,
    random_state=RANDOM_STATE
)

X_reduced = svd.fit_transform(X_informative)

print(f"Reduced matrix: {X_reduced.shape}")

print(
    "Explained variance ratio: "
    f"{svd.explained_variance_ratio_.sum():.4f}"
)

# Normalize reduced vectors.
normalizer = Normalizer(copy=False)
X_reduced = normalizer.fit_transform(X_reduced)

# --------------------------------------------------
# 3. KMEANS CLUSTERING
# --------------------------------------------------

print("\nTraining standard KMeans...")

model = KMeans(
    n_clusters=N_CLUSTERS,
    init="k-means++",
    n_init=10,
    max_iter=300,
    random_state=RANDOM_STATE,
    algorithm="lloyd"
)

labels = model.fit_predict(X_reduced)

df["cluster_id"] = -1
df.loc[informative_mask, "cluster_id"] = labels

print("Clustering completed.")

# --------------------------------------------------
# 4. CLUSTER SUMMARY
# --------------------------------------------------

print("\nCLUSTER SUMMARY")
print("=" * 65)

feature_names = vectorizer.get_feature_names_out()

summary = []

for cluster_id in range(N_CLUSTERS):

    cluster_mask = labels == cluster_id
    cluster_size = int(cluster_mask.sum())

    distances = np.linalg.norm(
        X_reduced[cluster_mask] - model.cluster_centers_[cluster_id],
        axis=1
    )

    nearest_indices = np.where(cluster_mask)[0][
        np.argsort(distances)[:5]
    ]

    summary.append({
        "cluster_id": cluster_id,
        "reviews": cluster_size,
        "percentage": cluster_size / len(labels) * 100,
        "top_terms": "",
        "representative_reviews": " | ".join(
            texts.iloc[informative_mask.nonzero()[0][i]][:180]
            for i in nearest_indices
        )
    })

    print(f"\nCLUSTER {cluster_id}")
    print(f"Reviews: {cluster_size:,}")
    print(f"Percentage: {cluster_size / len(labels) * 100:.2f}%")

    center = model.cluster_centers_[cluster_id]

# Project cluster center back into TF-IDF feature space
    original_center = svd.inverse_transform(center.reshape(1, -1))[0]

    top_feature_indices = np.argsort(
        original_center
    )[-15:][::-1]

    top_terms = [
        feature_names[i]
        for i in top_feature_indices
    ]

    print("Top complaint terms:")
    print(", ".join(top_terms))

    summary[-1]["top_terms"] = ", ".join(top_terms)

    print("Representative reviews:")

    for index in nearest_indices:
        original_index = informative_mask.nonzero()[0][index]
        row = df.iloc[original_index]

        print(f"\nApp: {row['app']}")
        print(f"Rating: {row['score']}")
        print(f"Review: {row['cleaned_text'][:250]}")

summary_df = pd.DataFrame(summary)

# --------------------------------------------------
# 5. SAVE OUTPUTS
# --------------------------------------------------

df.to_parquet(OUTPUT_PATH, index=False)
summary_df.to_csv(REPORT_PATH, index=False)

print("\nSaved clustered dataset:", OUTPUT_PATH)
print("Saved cluster report:", REPORT_PATH)

print("\nV3 EXPERIMENT COMPLETED.")
