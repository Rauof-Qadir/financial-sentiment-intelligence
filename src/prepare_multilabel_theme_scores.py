
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

INPUT_PATH = "data/processed/semantic_complaint_classifications.parquet"

EMBEDDINGS_PATH = "data/processed/core_complaint_embeddings.npy"
IDS_PATH = "data/processed/core_complaint_embedding_ids.parquet"

SCORES_PATH = "data/processed/multilabel_theme_scores.parquet"
SUMMARY_PATH = "reports/multilabel_theme_score_distribution.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 256

THEMES = {
    "App performance": (
        "The banking app crashes, freezes, loads slowly, "
        "shows technical errors, or stops working."
    ),
    "Login/authentication": (
        "The customer cannot log in or sign in, or has problems "
        "with passwords, PINs, OTP codes, fingerprints, "
        "verification, or authentication."
    ),
    "Account restriction": (
        "The customer's bank account is blocked, frozen, "
        "suspended, restricted, unexpectedly closed, "
        "or their money is held by the bank."
    ),
    "Fraud/security": (
        "The customer reports fraud, scams, stolen money, "
        "unauthorized transactions, identity theft, "
        "or security vulnerabilities."
    ),
    "Money transfer": (
        "The customer has problems sending, receiving, "
        "transferring, depositing, or withdrawing money."
    ),
    "Customer support": (
        "The customer complains about poor customer service, "
        "unresponsive support agents, live chat, delayed replies, "
        "or inability to contact the bank."
    ),
    "Card/payment": (
        "The customer has problems with debit cards, credit cards, "
        "declined card transactions, contactless payments, "
        "or making payments."
    ),
    "Features/usability": (
        "The customer complains about confusing navigation, "
        "poor interface design, missing features, difficult "
        "workflows, unwanted features, or poor app usability."
    ),
    "Trading/investment": (
        "The customer has problems buying or selling investments, "
        "trading, cryptocurrency, investment prices, "
        "market orders, or investment execution."
    ),
}


def load_or_create_embeddings(model, texts, review_ids):
    cache_exists = (
        os.path.exists(EMBEDDINGS_PATH)
        and os.path.exists(IDS_PATH)
    )

    if cache_exists:
        print("\nChecking existing embedding cache...")

        cached_ids = pd.read_parquet(IDS_PATH)["review_id"].astype(str)
        current_ids = pd.Series(review_ids).astype(str)

        if cached_ids.equals(current_ids):
            embeddings = np.load(EMBEDDINGS_PATH)

            if len(embeddings) == len(texts):
                print("Valid embedding cache found.")
                return embeddings

        print("Cache does not match current data. Rebuilding...")

    print("\nEncoding customer reviews...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    embeddings = embeddings.astype(np.float32)

    os.makedirs("data/processed", exist_ok=True)

    np.save(EMBEDDINGS_PATH, embeddings)

    pd.DataFrame({
        "review_id": pd.Series(review_ids).astype(str)
    }).to_parquet(IDS_PATH, index=False)

    print("Review embeddings cached successfully.")

    return embeddings


def main():
    print("Loading complaint dataset...")
    df = pd.read_parquet(INPUT_PATH)

    df = df[df["cleaned_text"].fillna("").str.strip() != ""].copy()
    df = df.reset_index(drop=True)

    print("Usable reviews:", len(df))

    model = SentenceTransformer(MODEL_NAME)

    review_ids = df["review_id"].tolist()
    texts = df["cleaned_text"].fillna("").astype(str).tolist()

    review_embeddings = load_or_create_embeddings(
        model,
        texts,
        review_ids
    )

    theme_names = list(THEMES.keys())
    descriptions = list(THEMES.values())

    print("\nEncoding theme descriptions...")

    theme_embeddings = model.encode(
        descriptions,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    print("\nCalculating theme similarity scores...")

    similarities = review_embeddings @ theme_embeddings.T

    score_data = {"review_id": review_ids}

    for index, theme in enumerate(theme_names):
        column = "score__" + theme.replace("/", "_").replace(" ", "_")
        score_data[column] = similarities[:, index]

    scores_df = pd.DataFrame(score_data)

    os.makedirs("reports", exist_ok=True)

    scores_df.to_parquet(SCORES_PATH, index=False)

    print("\nGenerating score distribution report...")

    summary = []

    for index, theme in enumerate(theme_names):
        values = similarities[:, index]

        summary.append({
            "theme": theme,
            "mean": np.mean(values),
            "min": np.min(values),
            "p10": np.percentile(values, 10),
            "p25": np.percentile(values, 25),
            "median": np.median(values),
            "p75": np.percentile(values, 75),
            "p90": np.percentile(values, 90),
            "max": np.max(values),
        })

    summary_df = pd.DataFrame(summary)

    numeric_columns = summary_df.select_dtypes(
        include="number"
    ).columns

    summary_df[numeric_columns] = summary_df[numeric_columns].round(4)

    summary_df = summary_df.sort_values(
        "mean",
        ascending=False
    )

    summary_df.to_csv(SUMMARY_PATH, index=False)

    print("\nTHEME SIMILARITY DISTRIBUTIONS")
    print("=" * 85)
    print(summary_df.to_string(index=False))

    print("\nSaved:")
    print(SCORES_PATH)
    print(SUMMARY_PATH)
    print(EMBEDDINGS_PATH)
    print(IDS_PATH)


if __name__ == "__main__":
    main()
