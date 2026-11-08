
import pandas as pd
import joblib

from sklearn.metrics import confusion_matrix, classification_report

DATA_PATH = "data/processed/banking_reviews_with_splits.parquet"
MODEL_PATH = "models/banking_review_best_model.joblib"

LABELS = ["negative", "neutral", "positive"]

print("\nLoading validation data...")

df = pd.read_parquet(DATA_PATH)

df = df[
    (df["split"] == "validation")
    & (df["has_cleaned_text"])
    & (df["rating_sentiment"].notna())
].copy()

print("Validation records:", len(df))

X = df["cleaned_text"]
y_true = df["rating_sentiment"]

print("\nLoading best model...")

bundle = joblib.load(MODEL_PATH)

vectorizer = bundle["vectorizer"]
model = bundle["classifier"]

X_vectorized = vectorizer.transform(X)
y_pred = model.predict(X_vectorized)

label_mapping = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

y_pred = pd.Series(y_pred).map(label_mapping).to_numpy()

print("\nConfusion Matrix:")
print(
    pd.DataFrame(
        confusion_matrix(
            y_true,
            y_pred,
            labels=LABELS
        ),
        index=[f"Actual_{x}" for x in LABELS],
        columns=[f"Predicted_{x}" for x in LABELS]
    )
)

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0
    )
)

df["predicted_sentiment"] = y_pred

print("\nNeutral review prediction distribution:")

neutral_reviews = df[df["rating_sentiment"] == "neutral"]

print(
    neutral_reviews["predicted_sentiment"]
    .value_counts()
)

print("\nMisclassified neutral examples:")

errors = neutral_reviews[
    neutral_reviews["predicted_sentiment"] != "neutral"
]

for _, row in errors.head(15).iterrows():
    print("\nActual:", row["rating_sentiment"])
    print("Predicted:", row["predicted_sentiment"])
    print("Review:", row["review_text"][:500])
    print("-" * 70)

print("\nERROR ANALYSIS COMPLETED.")
