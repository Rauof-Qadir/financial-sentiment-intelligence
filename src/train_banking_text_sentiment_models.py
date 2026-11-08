
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

DATA_PATH = "data/processed/banking_reviews_with_splits.parquet"
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["negative", "neutral", "positive"]

print("\nLoading banking reviews...")

df = pd.read_parquet(DATA_PATH)

# Use existing text annotations as the target.
df = df[
    df["has_cleaned_text"]
    & df["review_sentiment_label"].isin(LABELS)
].copy()

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()

print("\nDataset sizes:")
print("Training:", len(train_df))
print("Validation:", len(val_df))

print("\nTraining labels:")
print(train_df["review_sentiment_label"].value_counts())

print("\nValidation labels:")
print(val_df["review_sentiment_label"].value_counts())

X_train = train_df["cleaned_text"]
y_train = train_df["review_sentiment_label"]

X_val = val_df["cleaned_text"]
y_val = val_df["review_sentiment_label"]

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.95,
    max_features=30000,
    sublinear_tf=True,
    dtype=np.float32
)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

print("\nTF-IDF feature information:")
print("Training matrix:", X_train_vec.shape)
print("Validation matrix:", X_val_vec.shape)
print("Vocabulary size:", len(vectorizer.vocabulary_))


models = {
    "linear_svm": LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=42,
        dual="auto",
        max_iter=3000
    ),

    "sgd_logistic": SGDClassifier(
        loss="log_loss",
        class_weight="balanced",
        max_iter=1000,
        tol=1e-3,
        random_state=42,
        n_jobs=-1
    )
}

results = []
trained_models = {}

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    model.fit(X_train_vec, y_train)

    predictions = model.predict(X_val_vec)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_val, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_val, predictions
        ),
        "macro_precision": precision_score(
            y_val, predictions,
            average="macro",
            zero_division=0
        ),
        "macro_recall": recall_score(
            y_val, predictions,
            average="macro",
            zero_division=0
        ),
        "macro_f1": f1_score(
            y_val, predictions,
            average="macro",
            zero_division=0
        ),
        "weighted_f1": f1_score(
            y_val, predictions,
            average="weighted",
            zero_division=0
        )
    }

    results.append(metrics)
    trained_models[model_name] = model

    print(f"\n{model_name} classification report:")
    print(
        classification_report(
            y_val,
            predictions,
            labels=LABELS,
            zero_division=0
        )
    )


comparison = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(comparison.to_string(index=False))

comparison.to_csv(
    REPORT_DIR / "banking_text_annotation_model_comparison.csv",
    index=False
)

best_model_name = comparison.sort_values(
    "macro_f1",
    ascending=False
).iloc[0]["model"]

best_model = trained_models[best_model_name]

print("\nSelected model:", best_model_name)

model_bundle = {
    "vectorizer": vectorizer,
    "classifier": best_model,
    "model_name": best_model_name,
    "target_column": "review_sentiment_label",
    "labels": LABELS
}

joblib.dump(
    model_bundle,
    MODEL_DIR / "banking_text_annotation_best_model.joblib"
)

print("\nBest model saved successfully.")

print("\nBANKING TEXT ANNOTATION MODEL TRAINING COMPLETED.")
