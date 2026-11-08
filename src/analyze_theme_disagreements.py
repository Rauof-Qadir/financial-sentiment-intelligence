
import os
import pandas as pd

SEMANTIC_PATH = "data/processed/semantic_complaint_classifications.parquet"
KEYWORD_PATH = "reports/complaint_cluster_rule_validation.csv"

SUMMARY_PATH = "reports/theme_disagreement_summary.csv"
SAMPLES_PATH = "reports/theme_disagreement_samples.csv"

SAMPLES_PER_COMBINATION = 10


def main():
    print("Loading semantic classifications...")
    semantic = pd.read_parquet(SEMANTIC_PATH)

    print("Loading keyword classifications...")
    keyword = pd.read_csv(KEYWORD_PATH)

    required_semantic = {
        "review_id",
        "app",
        "score",
        "review_text",
        "semantic_primary_theme",
        "semantic_similarity",
        "similarity_margin",
        "low_confidence",
    }

    required_keyword = {"review_id", "primary_theme"}

    if not required_semantic.issubset(semantic.columns):
        raise ValueError(
            f"Missing semantic columns: "
            f"{required_semantic - set(semantic.columns)}"
        )

    if not required_keyword.issubset(keyword.columns):
        raise ValueError(
            f"Missing keyword columns: "
            f"{required_keyword - set(keyword.columns)}"
        )

    # Normalize IDs before merging.
    semantic["review_id"] = semantic["review_id"].astype(str)
    keyword["review_id"] = keyword["review_id"].astype(str)

    if semantic["review_id"].duplicated().any():
        raise ValueError("Duplicate review_id values in semantic data.")

    if keyword["review_id"].duplicated().any():
        raise ValueError("Duplicate review_id values in keyword data.")

    print("\nMerging predictions...")
    df = semantic.merge(
        keyword[["review_id", "primary_theme"]],
        on="review_id",
        how="inner",
        validate="one_to_one"
    )

    print("Matched reviews:", len(df))

    df = df.rename(columns={
        "primary_theme": "keyword_theme"
    })

    # Find disagreements.
    disagreements = df[
        df["keyword_theme"] != df["semantic_primary_theme"]
    ].copy()

    print("Disagreements:", len(disagreements))

    # Summarize every disagreement combination.
    summary = (
        disagreements
        .groupby(
            ["keyword_theme", "semantic_primary_theme"]
        )
        .size()
        .reset_index(name="review_count")
        .sort_values("review_count", ascending=False)
    )

    summary["percentage_of_disagreements"] = (
        summary["review_count"] / max(len(disagreements), 1) * 100
    ).round(2)

    summary.to_csv(SUMMARY_PATH, index=False)

    # Automatically select representative examples.
    selected_columns = [
        "review_id",
        "app",
        "score",
        "review_text",
        "keyword_theme",
        "semantic_primary_theme",
        "semantic_secondary_theme",
        "semantic_similarity",
        "second_similarity",
        "similarity_margin",
        "low_confidence",
    ]

    samples = []

    for _, group in disagreements.groupby(
        ["keyword_theme", "semantic_primary_theme"]
    ):
        sample = group.sample(
            n=min(SAMPLES_PER_COMBINATION, len(group)),
            random_state=42
        )
        samples.append(sample)

    if samples:
        sample_df = pd.concat(samples, ignore_index=True)
        sample_df = sample_df[selected_columns]
        sample_df.to_csv(SAMPLES_PATH, index=False)

    print("\nDISAGREEMENT ANALYSIS")
    print("=" * 65)
    print("Matched reviews:", len(df))
    print("Agreements:", len(df) - len(disagreements))
    print("Disagreements:", len(disagreements))

    print("\nTOP DISAGREEMENT COMBINATIONS")
    print(summary.head(15).to_string(index=False))

    print("\nSaved reports:")
    print(SUMMARY_PATH)
    print(SAMPLES_PATH)


if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    main()
