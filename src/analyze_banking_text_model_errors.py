
import pandas as pd
import joblib

from sklearn.metrics import confusion_matrix, classification_report

DATA_PATH = "data/processed/banking_reviews_with_splits.parquet"
MODEL_PATH = "models/banking_text_annotation_best_model.joblib"

LABELS = ["negative", "neutral", "positive"]

print("\nLoading validation data...")

df = pd.read_parquet(DATA_PATH)

df = df[
    (df["split"] == "validation")
    & df["has_cleaned_text"]
    & df["review_sentiment_label"].isin(LABELS)
].copy()

print("Validation records:", len(df))

print("\nLoading best model...")

bundle = joblib.load(MODEL_PATH)

vectorizer = bundle["vectorizer"]
model = bundle["classifier"]

X = vectorizer.transform(df["cleaned_text"])
y_true = df["review_sentiment_label"]

y_pred = model.predict(X)

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=LABELS
)

print(
    pd.DataFrame(
        cm,
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

errors = df[
    df["review_sentiment_label"]
    != df["predicted_sentiment"]
].copy()

print("\nTotal incorrect predictions:", len(errors))
print("Error percentage:", round(len(errors) / len(df) * 100, 2), "%")

print("\nError distribution by actual class:")

for label in LABELS:

    class_errors = errors[
        errors["review_sentiment_label"] == label
    ]

    print(f"\nActual class: {label}")
    print("Total errors:", len(class_errors))

    print(
        class_errors["predicted_sentiment"]
        .value_counts()
    )

print("\nMisclassified examples:")

for label in LABELS:

    print(f"\n{'=' * 25} {label.upper()} {'=' * 25}")

    class_errors = errors[
        errors["review_sentiment_label"] == label
    ]

    for _, row in class_errors.head(5).iterrows():

        print("\nActual:", row["review_sentiment_label"])
        print("Predicted:", row["predicted_sentiment"])
        print("Review:", row["review_text"][:400])
        print("-" * 60)

errors.to_csv(
    "reports/banking_text_model_errors.csv",
    index=False
)

print("\nError report saved.")
print("\nBANKING TEXT MODEL ERROR ANALYSIS COMPLETED.")
