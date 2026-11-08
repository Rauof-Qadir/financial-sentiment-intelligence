
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

INPUT_PATH = "data/processed/core_complaints_clustered_k10_v3.parquet"
OUTPUT_PATH = "data/processed/semantic_complaint_classifications.parquet"
SUMMARY_PATH = "reports/semantic_complaint_theme_summary.csv"
COMPARISON_PATH = "reports/keyword_vs_semantic_comparison.csv"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Business-oriented descriptions, not just short category names.
THEMES = {
    "Account restriction": (
        "The customer cannot access their bank account, "
        "account is frozen, blocked, suspended or closed, "
        "or their money is held due to account restrictions."
    ),
    "Fraud/security": (
        "The customer reports fraud, scams, stolen money, "
        "unauthorized transactions, identity theft or security problems."
    ),
    "Money transfer": (
        "The customer has problems sending, receiving, transferring, "
        "depositing or withdrawing money."
    ),
    "Login/authentication": (
        "The customer cannot log in or sign in, has password, "
        "passcode, fingerprint, verification or authentication problems."
    ),
    "Customer support": (
        "The customer complains about poor customer service, "
        "unresponsive support agents, live chat, delayed replies "
        "or inability to contact the bank."
    ),
    "Card/payment": (
        "The customer has problems with debit cards, credit cards, "
        "card payments, declined transactions, contactless payments, "
        "fees or subscriptions."
    ),
    "App performance": (
        "The banking application crashes, freezes, loads slowly, "
        "has bugs, errors, compatibility problems or stops working."
    ),
    "Other/unclear": (
        "The review describes a banking complaint that does not "
        "clearly belong to the other categories."
    ),
}

# Initial similarity thresholds; these are not calibrated probabilities.
MIN_SIMILARITY = 0.35
MIN_MARGIN = 0.03
BATCH_SIZE = 256


def main():
    print("Loading complaint dataset...")
    df = pd.read_parquet(INPUT_PATH)

    df["cleaned_text"] = df["cleaned_text"].fillna("").astype(str)

    print("Total reviews:", len(df))

    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    theme_names = list(THEMES.keys())
    theme_descriptions = list(THEMES.values())

    print("Encoding complaint themes...")
    theme_embeddings = model.encode(
        theme_descriptions,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    texts = df["cleaned_text"].tolist()
    primary_themes = []
    secondary_themes = []
    top_scores = []
    second_scores = []
    margins = []
    low_confidence_flags = []

    print("\nEncoding and classifying reviews...")

    for start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[start:start + BATCH_SIZE]

        embeddings = model.encode(
            batch_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=BATCH_SIZE
        )

        similarities = np.asarray(embeddings) @ theme_embeddings.T

        # Sort themes by similarity for each review.
        order = np.argsort(similarities, axis=1)[:, ::-1]

        for i, indices in enumerate(order):
            first_idx = indices[0]
            second_idx = indices[1]

            first_score = float(similarities[i, first_idx])
            second_score = float(similarities[i, second_idx])
            margin = first_score - second_score

            low_confidence = (
                first_score < MIN_SIMILARITY
                or margin < MIN_MARGIN
            )

            primary = theme_names[first_idx]
            secondary = theme_names[second_idx]

            if low_confidence:
                primary = "Other/unclear"

            primary_themes.append(primary)
            secondary_themes.append(secondary)
            top_scores.append(first_score)
            second_scores.append(second_score)
            margins.append(margin)
            low_confidence_flags.append(low_confidence)

        processed = min(start + len(batch_texts), len(texts))
        print(f"Processed {processed:,}/{len(texts):,}")

    df["semantic_primary_theme"] = primary_themes
    df["semantic_secondary_theme"] = secondary_themes
    df["semantic_similarity"] = top_scores
    df["second_similarity"] = second_scores
    df["similarity_margin"] = margins
    df["low_confidence"] = low_confidence_flags

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    df.to_parquet(OUTPUT_PATH, index=False)

    summary = (
        df.groupby("semantic_primary_theme")
        .agg(
            total_reviews=("cleaned_text", "size"),
            average_similarity=("semantic_similarity", "mean"),
            low_confidence_reviews=("low_confidence", "sum")
        )
        .sort_values("total_reviews", ascending=False)
    )

    summary["percentage"] = (
        summary["total_reviews"] / len(df) * 100
    ).round(2)

    summary["average_similarity"] = (
        summary["average_similarity"].round(4)
    )

    summary.to_csv(SUMMARY_PATH)

    keyword_path = "reports/complaint_cluster_rule_validation.csv"

    if os.path.exists(keyword_path):
        keyword_df = pd.read_csv(keyword_path)

        comparison_df = df[["review_id", "semantic_primary_theme"]].merge(
            keyword_df[["review_id", "primary_theme"]],
            on="review_id",
            how="inner"
        )

        comparison = pd.crosstab(
            comparison_df["primary_theme"],
            comparison_df["semantic_primary_theme"],
            margins=True

            )

        comparison.to_csv(COMPARISON_PATH)

        print("\nKEYWORD VS SEMANTIC CLASSIFICATION COMPARISON saved:")

    else:

         print("\nKeyword report not found. Comparison skipped.")


    print("\nSEMANTIC CLASSIFICATION SUMMARY")
    print("=" * 60)
    print(summary.to_string())

    print("\nLow-confidence reviews:", sum(low_confidence_flags))
    print("Saved:", OUTPUT_PATH)
    print("Saved:", SUMMARY_PATH)
    print("Saved:", COMPARISON_PATH)


if __name__ == "__main__":
    main()
