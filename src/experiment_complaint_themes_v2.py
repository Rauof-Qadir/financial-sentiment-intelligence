
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_PATH = "data/processed/banking_complaints_core.parquet"

OUTPUT_PATH = (
    "data/processed/core_complaints_clustered_k12_v2.parquet"
)

REPORT_PATH = "reports/complaint_cluster_summary_k12_v2.csv"

N_CLUSTERS = 12
RANDOM_STATE = 42

# Generic sentiment words, not actual complaint themes
CUSTOM_STOPWORDS = [
    "bad",
    "terrible",
    "useless",
    "worst",
    "poor",
    "awful",
    "horrible",
    "disgusting",
    "pathetic",
    "excellent",
    "good",
    "great",
    "amazing",
    "brilliant",
    "app",
    "apps",
    "bank",
    "banking",
    "really",
    "absolutely",
    "totally",
    "very",
    "just",
    "thing",
    "things"
]

# --------------------------------------------------
# 1. Load complaint dataset
# --------------------------------------------------

print("\nLoading core complaints...")

df = pd.read_parquet(DATA_PATH)

df = df[
    df["cleaned_text"].notna()
    & df["cleaned_text"].str.strip().ne("")
].copy()

print("Usable complaints:", len(df))

texts = df["cleaned_text"].astype(str)

# --------------------------------------------------
# 2. Create improved TF-IDF features
# --------------------------------------------------

print("\nCreating improved TF-IDF features...")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.90,
    max_features=40000,
    stop_words="english",
    sublinear_tf=True,
    dtype="float32"
)

X = vectorizer.fit_transform(texts)

# Remove generic sentiment words from topic features
print("\nRemoving generic sentiment terms...")

feature_names = vectorizer.get_feature_names_out()

feature_mask = [
    not any(
        word in CUSTOM_STOPWORDS
        for word in term.split()
    )
    for term in feature_names
]

X = X[:, feature_mask]

terms = feature_names[feature_mask]

print("TF-IDF matrix shape:", X.shape)
print("Vocabulary size:", len(terms))

# --------------------------------------------------
# 3. Train MiniBatchKMeans
# --------------------------------------------------

print("\nTraining MiniBatchKMeans...")

model = MiniBatchKMeans(
    n_clusters=N_CLUSTERS,
    random_state=RANDOM_STATE,
    batch_size=2048,
    n_init=10,
    max_iter=100,
    reassignment_ratio=0.01
)

df["cluster_id"] = model.fit_predict(X)

print("Clustering completed.")

# --------------------------------------------------
# 4. Extract cluster terms
# --------------------------------------------------

print("\nTOP TERMS AND CLUSTER SIZES")

centers = model.cluster_centers_

summary = []

for cluster_id in range(N_CLUSTERS):

    cluster_mask = df["cluster_id"] == cluster_id
    cluster_size = int(cluster_mask.sum())

    top_indices = centers[cluster_id].argsort()[-15:][::-1]

    top_terms = [
        terms[index]
        for index in top_indices
    ]

    summary.append({
        "cluster_id": cluster_id,
        "review_count": cluster_size,
        "percentage": round(
            cluster_size / len(df) * 100, 2
        ),
        "top_terms": ", ".join(top_terms)
    })

    print("\n" + "=" * 65)
    print(f"CLUSTER {cluster_id}")
    print("Reviews:", cluster_size)
    print(
        "Percentage:",
        round(cluster_size / len(df) * 100, 2),
        "%"
    )
    print("Top terms:", ", ".join(top_terms))

# --------------------------------------------------
# 5. Extract representative reviews
# --------------------------------------------------

print("\nREPRESENTATIVE REVIEWS")

distances = model.transform(X)

for cluster_id in range(N_CLUSTERS):

    print("\n" + "=" * 65)
    print(f"CLUSTER {cluster_id} - SAMPLE REVIEWS")

    positions = [
        i for i, value in enumerate(df["cluster_id"])
        if value == cluster_id
    ]

    if not positions:
        continue

    nearest_positions = (
        distances[positions, cluster_id]
        .argsort()[:5]
    )

    for position in nearest_positions:

        row = df.iloc[positions[position]]

        print("\nApp:", row["app"])
        print("Rating:", row["score"])
        print("Review:", row["cleaned_text"][:350])

# --------------------------------------------------
# 6. Save results
# --------------------------------------------------

df.to_parquet(OUTPUT_PATH, index=False)

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    REPORT_PATH,
    index=False
)

print("\nSaved clustered dataset:", OUTPUT_PATH)
print("Saved cluster report:", REPORT_PATH)

print("\nCOMPLAINT THEME DISCOVERY V2 COMPLETED.")
