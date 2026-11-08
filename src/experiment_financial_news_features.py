
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
    classification_report,
)


ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT_DIR / "data" / "processed" / "financial_news_cleaned.parquet"
BASELINE_PATH = ROOT_DIR / "models" / "financial_news_best_model.joblib"
REPORT_DIR = ROOT_DIR / "reports"
MODEL_DIR = ROOT_DIR / "models"

LABEL_NAMES = ["negative", "neutral", "positive"]


# 1. Load validation dataset

df = pd.read_parquet(DATA_PATH)

train_df = df[
    (df["split"] == "train") &
    df["has_cleaned_text"]
]

val_df = df[
    (df["split"] == "validation") &
    df["has_cleaned_text"]
]

X_train = train_df["cleaned_text"]
y_train = train_df["label"]

X_val = val_df["cleaned_text"]
y_val = val_df["label"]


# 2. Load existing baseline

baseline = joblib.load(BASELINE_PATH)

baseline_predictions = baseline.predict(X_val)


def evaluate_model(name, predictions):

    return {
        "model": name,
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
            y_val, predictions, labels=[2], average="macro", zero_division=0
        ),
    }


results = []

results.append(
    evaluate_model("baseline_word_tfidf_svm", baseline_predictions)
)


# 3. Create word + character TF-IDF

combined_features = FeatureUnion([
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


# 4. Create improved model

improved_model = Pipeline([
    ("features", combined_features),
    (
        "classifier",
        LinearSVC(
            class_weight="balanced",
            random_state=42,
        ),
    ),
])


# 5. Train using training data only

print("\nTraining word + character TF-IDF SVM...")

improved_model.fit(X_train, y_train)

improved_predictions = improved_model.predict(X_val)


# 6. Evaluate improved model

results.append(
    evaluate_model("word_char_tfidf_svm", improved_predictions)
)


print("\nIMPROVED MODEL CLASSIFICATION REPORT")

print(
    classification_report(
        y_val,
        improved_predictions,
        labels=[0, 1, 2],
        target_names=LABEL_NAMES,
        zero_division=0,
    )
)


# 7. Compare models

results_df = pd.DataFrame(results)

print("\nMODEL EXPERIMENT COMPARISON")

print(results_df.to_string(index=False))

results_df.to_csv(
    REPORT_DIR / "financial_news_feature_experiments.csv",
    index=False,
)


# 8. Save experimental model separately

joblib.dump(
    improved_model,
    MODEL_DIR / "financial_news_word_char_experiment.joblib",
)

print("\nExperimental model saved.")
print("\nFEATURE EXPERIMENT COMPLETED")
