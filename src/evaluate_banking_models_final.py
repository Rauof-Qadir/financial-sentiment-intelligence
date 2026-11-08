
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

DATA_PATH = "data/processed/banking_reviews_with_splits.parquet"

LABELS = ["negative", "neutral", "positive"]

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("\nLoading test dataset...")

df = pd.read_parquet(DATA_PATH)

test_df = df[
    (df["split"] == "test")
    & (df["has_cleaned_text"])
].copy()

print("Test records:", len(test_df))

results = {}
all_confusion_matrices = {}

experiments = {
    "rating_derived": {
        "path": "models/banking_review_best_model.joblib",
        "target": "rating_sentiment"
    },
    "text_annotation": {
        "path": "models/banking_text_annotation_best_model.joblib",
        "target": "review_sentiment_label"
    }
}

for experiment_name, config in experiments.items():

    print(f"\n{'=' * 25}")
    print("Evaluating:", experiment_name)
    print(f"{'=' * 25}")

    target = config["target"]

    evaluation_df = test_df[
        test_df[target].isin(LABELS)
    ].copy()

    bundle = joblib.load(config["path"])

    vectorizer = bundle["vectorizer"]
    model = bundle["classifier"]

    X = vectorizer.transform(
        evaluation_df["cleaned_text"]
    )

    y_true = evaluation_df[target].to_numpy()
    y_pred = model.predict(X)

    # The rating-derived model uses numeric labels.
    if experiment_name == "rating_derived":

        label_mapping = bundle["label_mapping"]

        reverse_mapping = {
            value: key
            for key, value in label_mapping.items()
        }

        y_pred = pd.Series(y_pred).map(
            reverse_mapping
        ).to_numpy()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_precision": float(
            precision_score(
                y_true, y_pred,
                labels=LABELS,
                average="macro",
                zero_division=0
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true, y_pred,
                labels=LABELS,
                average="macro",
                zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true, y_pred,
                labels=LABELS,
                average="macro",
                zero_division=0
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true, y_pred,
                labels=LABELS,
                average="weighted",
                zero_division=0
            )
        ),
        "test_records": int(len(evaluation_df))
    }

    results[experiment_name] = metrics

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            zero_division=0
        )
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS
    )

    all_confusion_matrices[experiment_name] = (
        pd.DataFrame(
            cm,
            index=[f"Actual_{x}" for x in LABELS],
            columns=[f"Predicted_{x}" for x in LABELS]
        )
    )

    all_confusion_matrices[experiment_name].to_csv(
        REPORT_DIR / f"{experiment_name}_test_confusion_matrix.csv"
    )

    print("\nConfusion Matrix:")
    print(all_confusion_matrices[experiment_name])


with open(
    REPORT_DIR / "banking_final_test_metrics.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(results, f, indent=4)


comparison = pd.DataFrame(results).T

comparison.to_csv(
    REPORT_DIR / "banking_final_test_comparison.csv"
)

print("\nFINAL TEST COMPARISON")
print(comparison.round(4).to_string())

print("\nAll final evaluation reports saved.")
print("\nBANKING FINAL TEST EVALUATION COMPLETED.")
