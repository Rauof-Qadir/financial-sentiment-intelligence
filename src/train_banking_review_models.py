
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)


ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    ROOT_DIR / "data" / "processed"
    / "banking_reviews_with_splits.parquet"
)

REPORT_DIR = ROOT_DIR / "reports"
MODEL_DIR = ROOT_DIR / "models"

LABEL_NAMES = ["negative", "neutral", "positive"]

# 1. Load dataset

print("\nLoading banking reviews...")

df = pd.read_parquet(DATA_PATH)

# Exclude records without usable text or target
df = df[
    df["has_cleaned_text"]
    & df["rating_sentiment"].notna()
].copy()

# 2. Create chronological datasets

train_df = df[df["split"] == "train"]
val_df = df[df["split"] == "validation"]

X_train = train_df["cleaned_text"]
y_train = train_df["rating_sentiment"]

X_val = val_df["cleaned_text"]
y_val = val_df["rating_sentiment"]

print("\nDataset sizes:")
print("Training:", len(X_train))
print("Validation:", len(X_val))

print("\nTraining labels:")
print(y_train.value_counts())

# 3. Convert labels to numeric values

label_mapping = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

y_train = y_train.map(label_mapping)
y_val = y_val.map(label_mapping)

# 4. TF-IDF feature extraction

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.95,
    max_features=30000,
    sublinear_tf=True,
    dtype=np.float32,
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_val_tfidf = vectorizer.transform(X_val)

print("\nTF-IDF feature information:")
print("Training matrix:", X_train_tfidf.shape)
print("Validation matrix:", X_val_tfidf.shape)
print("Vocabulary size:", len(vectorizer.vocabulary_))

# 5. Define models

models = {
    "linear_svm": LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=42,
        dual="auto",
        max_iter=3000,
    ),

    "sgd_logistic": SGDClassifier(
        loss="log_loss",
        class_weight="balanced",
        max_iter=1000,
        tol=1e-3,
        random_state=42,
        n_jobs=-1,
    ),
}

results = []
trained_models = {}

# 6. Train and evaluate

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_val_tfidf)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_val, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_val, predictions
        ),
        "macro_precision": precision_score(
            y_val, predictions,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            y_val, predictions,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_val, predictions,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_val, predictions,
            average="weighted",
            zero_division=0,
        ),
    }

    results.append(metrics)
    trained_models[model_name] = model

    print(f"\n{model_name} classification report:")

    print(
        classification_report(
            y_val,
            predictions,
            labels=[0, 1, 2],
            target_names=LABEL_NAMES,
            zero_division=0,
        )
    )

# 7. Compare models

results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")

print(
    results_df.sort_values(
        "macro_f1",
        ascending=False
    ).to_string(index=False)
)

results_df.to_csv(
    REPORT_DIR / "banking_review_model_comparison.csv",
    index=False,
)

# 8. Select best model using validation Macro F1

best_model_name = results_df.loc[
    results_df["macro_f1"].idxmax(),
    "model",
]

best_model = trained_models[best_model_name]

print("\nSelected model:", best_model_name)

# 9. Save model and vectorizer together

model_bundle = {
    "vectorizer": vectorizer,
    "classifier": best_model,
    "label_mapping": label_mapping,
    "model_name": best_model_name,
}

joblib.dump(
    model_bundle,
    MODEL_DIR / "banking_review_best_model.joblib",
)

print("\nBest model saved successfully.")

print("\nBANKING REVIEW MODEL TRAINING COMPLETED.")
