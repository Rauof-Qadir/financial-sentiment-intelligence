
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT_DIR / "data" / "processed"
    / "financial_news_cleaned.parquet"
)

MODEL_PATH = (
    ROOT_DIR / "models"
    / "financial_news_best_word_char_svm.joblib"
)

REPORT_DIR = ROOT_DIR / "reports"

LABEL_NAMES = ["negative", "neutral", "positive"]

# 1. Load final model
print("Loading final selected model...")

model = joblib.load(MODEL_PATH)

# 2. Load untouched test dataset
print("Loading untouched test dataset...")

df = pd.read_parquet(DATA_PATH)

test_df = df[
    (df["split"] == "test") &
    (df["has_cleaned_text"])
].copy()

X_test = test_df["cleaned_text"]
y_test = test_df["label"]

print(f"\nTest records: {len(test_df)}")

# 3. Generate predictions
print("\nGenerating test predictions...")

predictions = model.predict(X_test)

# 4. Calculate metrics
metrics = {
    "model": "word_char_tfidf_linear_svm",
    "C": 0.5,
    "accuracy": accuracy_score(y_test, predictions),
    "macro_precision": precision_score(
        y_test, predictions, average="macro", zero_division=0
    ),
    "macro_recall": recall_score(
        y_test, predictions, average="macro", zero_division=0
    ),
    "macro_f1": f1_score(
        y_test, predictions, average="macro", zero_division=0
    ),
    "weighted_f1": f1_score(
        y_test, predictions, average="weighted", zero_division=0
    ),
}

print("\nFINAL TEST METRICS")

for key, value in metrics.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

# 5. Per-class performance
print("\nCLASSIFICATION REPORT")

report = classification_report(
    y_test,
    predictions,
    labels=[0, 1, 2],
    target_names=LABEL_NAMES,
    zero_division=0,
)

print(report)

# 6. Confusion matrix
cm = confusion_matrix(
    y_test,
    predictions,
    labels=[0, 1, 2],
)

cm_df = pd.DataFrame(
    cm,
    index=LABEL_NAMES,
    columns=LABEL_NAMES,
)

print("\nCONFUSION MATRIX")
print(cm_df)

# 7. Save results
with open(
    REPORT_DIR / "financial_news_final_test_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(metrics, file, indent=4)

cm_df.to_csv(
    REPORT_DIR / "financial_news_final_test_confusion_matrix.csv"
)

# 8. Save predictions for error analysis
test_df["predicted_label"] = predictions

test_df["actual_sentiment"] = test_df["label"].map({
    0: "negative",
    1: "neutral",
    2: "positive",
})

test_df["predicted_sentiment"] = test_df["predicted_label"].map({
    0: "negative",
    1: "neutral",
    2: "positive",
})

test_df["is_correct"] = (
    test_df["label"] == test_df["predicted_label"]
)

test_df.to_csv(
    REPORT_DIR / "financial_news_test_predictions.csv",
    index=False,
)

print("\nFINAL TEST EVALUATION COMPLETED.")
print("All test reports saved.")
