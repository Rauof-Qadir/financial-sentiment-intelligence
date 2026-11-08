
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT_DIR / "data" / "processed" / "financial_news_cleaned.parquet"
REPORT_DIR = ROOT_DIR / "reports"
MODEL_DIR = ROOT_DIR / "models"

df = pd.read_parquet(DATA_PATH)

train_df = df[
    (df["split"] == "train") & df["has_cleaned_text"]
]

val_df = df[
    (df["split"] == "validation") & df["has_cleaned_text"]
]

X_train = train_df["cleaned_text"]
y_train = train_df["label"]

X_val = val_df["cleaned_text"]
y_val = val_df["label"]


def build_model(c_value):

    features = FeatureUnion([
        (
            "word_tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                max_features=30000,
                sublinear_tf=True,
            ),
        ),
        (
            "char_tfidf",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=2,
                max_features=50000,
                sublinear_tf=True,
            ),
        ),
    ])

    return Pipeline([
        ("features", features),
        (
            "classifier",
            LinearSVC(
                C=c_value,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])


results = []
trained_models = {}

for c_value in [0.5, 1.0, 2.0]:

    print(f"\nTraining Linear SVM with C={c_value}...")

    model = build_model(c_value)
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    metrics = {
        "C": c_value,
        "accuracy": accuracy_score(y_val, predictions),
        "macro_precision": precision_score(
            y_val, predictions, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_val, predictions, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_val, predictions, average="macro", zero_division=0
        ),
        "positive_recall": recall_score(
            y_val, predictions, labels=[2],
            average="macro", zero_division=0
        ),
    }

    results.append(metrics)
    trained_models[c_value] = model


results_df = pd.DataFrame(results)

print("\nSVM C PARAMETER COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv(
    REPORT_DIR / "financial_news_svm_c_experiment.csv",
    index=False,
)

best_row = results_df.loc[results_df["macro_f1"].idxmax()]
best_c = float(best_row["C"])

joblib.dump(
    trained_models[best_c],
    MODEL_DIR / "financial_news_best_word_char_svm.joblib",
)

print(f"\nBest C based on validation Macro F1: {best_c}")
print("Best experimental model saved.")
