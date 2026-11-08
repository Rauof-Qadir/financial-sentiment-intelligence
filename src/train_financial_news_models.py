
from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline


# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT_DIR / "data" / "processed" / "financial_news_cleaned.parquet"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TEXT_COLUMN = "cleaned_text"
TARGET_COLUMN = "label"

LABEL_NAMES = ["negative", "neutral", "positive"]


# --------------------------------------------------
# 2. Load and prepare dataset
# --------------------------------------------------

print("\nLoading financial news dataset...")

df = pd.read_parquet(DATA_PATH)

# Exclude records with empty text from model training
df = df[df["has_cleaned_text"]].copy()

train_df = df[df["split"] == "train"]
val_df = df[df["split"] == "validation"]
test_df = df[df["split"] == "test"]

X_train = train_df[TEXT_COLUMN]
y_train = train_df[TARGET_COLUMN]

X_val = val_df[TEXT_COLUMN]
y_val = val_df[TARGET_COLUMN]

X_test = test_df[TEXT_COLUMN]
y_test = test_df[TARGET_COLUMN]

print("Training records:", len(X_train))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))


# --------------------------------------------------
# 3. Define TF-IDF configuration
# --------------------------------------------------

def create_tfidf_model(classifier):

    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=False,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                max_features=30000,
                sublinear_tf=True,
            ),
        ),
        ("classifier", classifier),
    ])


# --------------------------------------------------
# 4. Define classification models
# --------------------------------------------------

models = {
    "logistic_regression": create_tfidf_model(
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )
    ),

    "linear_svm": create_tfidf_model(
        LinearSVC(
            class_weight="balanced",
            random_state=42,
        )
    ),
}


# --------------------------------------------------
# 5. Train and evaluate on validation data
# --------------------------------------------------

results = []
trained_models = {}

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    metrics = {
        "model": model_name,
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
        "weighted_f1": f1_score(
            y_val, predictions, average="weighted", zero_division=0
        ),
    }

    results.append(metrics)
    trained_models[model_name] = model

    print("\nValidation metrics:")
    print(json.dumps(metrics, indent=4))

    print("\nClassification report:")
    print(
        classification_report(
            y_val,
            predictions,
            labels=[0, 1, 2],
            target_names=LABEL_NAMES,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_val, predictions, labels=[0, 1, 2]))


# --------------------------------------------------
# 6. Compare models
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="macro_f1",
    ascending=False,
)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv(
    REPORT_DIR / "financial_news_model_comparison.csv",
    index=False,
)


# --------------------------------------------------
# 7. Select best model using validation Macro F1
# --------------------------------------------------

best_model_name = results_df.iloc[0]["model"]
best_model = trained_models[best_model_name]

print("\nSelected model:", best_model_name)


# --------------------------------------------------
# 8. Save selected model
# --------------------------------------------------

model_path = MODEL_DIR / "financial_news_best_model.joblib"

joblib.dump(best_model, model_path)

print("\nSaved model:", model_path)

print("\nMODEL TRAINING COMPLETED")
