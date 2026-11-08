
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_PATH = "data/processed/banking_complaints_core.parquet"

CUSTOM_STOPWORDS = [
    "bad", "terrible", "useless", "worst", "poor", "awful",
    "horrible", "disgusting", "pathetic", "excellent", "good",
    "great", "amazing", "brilliant", "app", "apps", "bank",
    "banking", "really", "absolutely", "totally", "very",
    "just", "thing", "things"
]

print("Loading core complaints...")
df = pd.read_parquet(DATA_PATH)

df = df[
    df["cleaned_text"].notna()
    & df["cleaned_text"].str.strip().ne("")
].copy()

texts = df["cleaned_text"].astype(str)

print(f"Usable complaints: {len(df):,}")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.90,
    max_features=40000,
    stop_words="english",
    sublinear_tf=True,
    dtype=np.float32
)

print("\nCreating original TF-IDF features...")
X = vectorizer.fit_transform(texts)

feature_names = vectorizer.get_feature_names_out()

feature_mask = np.array([
    not any(word in CUSTOM_STOPWORDS for word in term.split())
    for term in feature_names
])

X_filtered = X[:, feature_mask]

original_features = np.diff(X.indptr)
remaining_features = np.diff(X_filtered.indptr)

df["original_feature_count"] = original_features
df["remaining_feature_count"] = remaining_features

zero_feature_count = int((remaining_features == 0).sum())
informative_count = int((remaining_features > 0).sum())

print("\nFEATURE COVERAGE AUDIT")
print("=" * 50)

print(f"Original vocabulary: {X.shape[1]:,}")
print(f"Remaining vocabulary: {X_filtered.shape[1]:,}")
print(f"Original non-zero feature rows: {(original_features > 0).sum():,}")
print(f"Zero-feature rows after filtering: {zero_feature_count:,}")
print(f"Informative rows after filtering: {informative_count:,}")
print(
    f"Zero-feature percentage: "
    f"{zero_feature_count / len(df) * 100:.2f}%"
)

print("\nSAMPLE REVIEWS WITH ZERO FEATURES")
print("=" * 50)

for _, row in df[remaining_features == 0].head(15).iterrows():
    print(f"\nRating: {row['score']}")
    print(f"Review: {row['cleaned_text'][:250]}")

print("\nSAMPLE REVIEWS WITH REMAINING FEATURES")
print("=" * 50)

for _, row in df[remaining_features > 0].head(15).iterrows():
    print(f"\nRemaining features: {row['remaining_feature_count']}")
    print(f"Review: {row['cleaned_text'][:250]}")

print("\nAUDIT COMPLETED.")
