
import pandas as pd

INPUT_PATH = "reports/theme_validation_with_weak_evidence.csv"
OUTPUT_PATH = "reports/theme_validation_evidence_summary.csv"

df = pd.read_csv(INPUT_PATH)

# Same mapping rules used in the previous mapping audit.
TOPIC_MAPPING = {
    "Support & Access": [
        "Customer_support",
        "Account_restriction",
        "Login_authentication",
    ],
    "Updates, Crashes & Performance": [
        "App_performance",
    ],
    "Fees & Service & Travel": [
        "Customer_support",
        "Card_payment",
        "Money_transfer",
    ],
    "Cards & Payments": [
        "Card_payment",
    ],
    "Login, Navigation & Functionality Issues": [
        "Login_authentication",
        "Features_usability",
    ],
    "Transactions, Features & Usability": [
        "Features_usability",
        "Money_transfer",
        "Card_payment",
    ],
    "Money Transfers & Payment Management": [
        "Money_transfer",
        "Card_payment",
    ],
    "Undefined": [],
}

def check_mapping(row):
    allowed = TOPIC_MAPPING.get(
        row["topic_label_ALL"], []
    )
    return row["candidate_theme"] in allowed

df["weak_topic_match"] = df.apply(
    check_mapping,
    axis=1
)

summary = (
    df.groupby(
        ["candidate_theme", "score_band"],
        observed=True
    )
    .agg(
        examples=("review_id", "size"),
        average_similarity=("similarity_score", "mean"),
        weak_topic_matches=("weak_topic_match", "sum"),
    )
    .reset_index()
)

summary["weak_match_rate"] = (
    summary["weak_topic_matches"] / summary["examples"]
)

summary.to_csv(OUTPUT_PATH, index=False)

print("TOTAL WEAK TOPIC MATCHES:",
      df["weak_topic_match"].sum())

print("TOTAL EXAMPLES:", len(df))

print(
    "WEAK MATCH RATE:",
    round(df["weak_topic_match"].mean() * 100, 2),
    "%"
)

print("\nTHEME VALIDATION EVIDENCE")
print("=" * 75)

for theme, group in summary.groupby("candidate_theme"):
    print(f"\n{theme}")
    print(group.to_string(index=False))

print("\nSaved:", OUTPUT_PATH)
