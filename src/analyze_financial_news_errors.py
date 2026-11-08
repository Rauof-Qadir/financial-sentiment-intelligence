
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix


# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT_DIR / "data" / "processed" / "financial_news_cleaned.parquet"
MODEL_PATH = ROOT_DIR / "models" / "financial_news_best_model.joblib"
REPORT_DIR = ROOT_DIR / "reports"

LABEL_NAMES = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


# Load validation data
df = pd.read_parquet(DATA_PATH)

val_df = df[
    (df["split"] == "validation") &
    (df["has_cleaned_text"])
].copy()


# Load selected model
model = joblib.load(MODEL_PATH)


# Generate predictions
X_val = val_df["cleaned_text"]
y_true = val_df["label"]

y_pred = model.predict(X_val)


# Add prediction results
val_df["predicted_label"] = y_pred

val_df["actual_sentiment"] = val_df["label"].map(LABEL_NAMES)
val_df["predicted_sentiment"] = val_df["predicted_label"].map(LABEL_NAMES)

val_df["is_correct"] = (
    val_df["label"] == val_df["predicted_label"]
)


# Extract incorrect predictions
errors = val_df[~val_df["is_correct"]].copy()

error_output = errors[
    [
        "sentence",
        "cleaned_text",
        "actual_sentiment",
        "predicted_sentiment",
    ]
]

error_output.to_csv(
    REPORT_DIR / "financial_news_validation_errors.csv",
    index=False,
)


# Save confusion matrix
matrix = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1, 2],
)

matrix_df = pd.DataFrame(
    matrix,
    index=["actual_negative", "actual_neutral", "actual_positive"],
    columns=["predicted_negative", "predicted_neutral", "predicted_positive"],
)

matrix_df.to_csv(
    REPORT_DIR / "financial_news_validation_confusion_matrix.csv"
)


# Print classification report
print("\nCLASSIFICATION REPORT")

print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=["negative", "neutral", "positive"],
        zero_division=0,
    )
)


# Print error statistics
print("\nERROR ANALYSIS")

print("Total validation records:", len(val_df))
print("Correct predictions:", val_df["is_correct"].sum())
print("Incorrect predictions:", len(errors))

print("\nActual vs predicted sentiment:")

print(
    pd.crosstab(
        errors["actual_sentiment"],
        errors["predicted_sentiment"],
    )
)


# Display positive-to-neutral mistakes
positive_misses = errors[
    (errors["actual_sentiment"] == "positive") &
    (errors["predicted_sentiment"] == "neutral")
]

print("\nPOSITIVE NEWS MISCLASSIFIED AS NEUTRAL")

for _, row in positive_misses.head(15).iterrows():
    print("\nActual:", row["actual_sentiment"])
    print("Predicted:", row["predicted_sentiment"])
    print("Sentence:", row["sentence"])


print("\nERROR ANALYSIS COMPLETED")
