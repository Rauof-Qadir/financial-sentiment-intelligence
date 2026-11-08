
import pandas as pd

INPUT_PATH = "reports/theme_validation_with_weak_evidence.csv"
OUTPUT_PATH = "reports/topic_to_theme_mapping_audit.csv"

df = pd.read_csv(INPUT_PATH)

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
    topic = row["topic_label_ALL"]
    theme = row["candidate_theme"]

    allowed_themes = TOPIC_MAPPING.get(topic, [])

    return theme in allowed_themes

df["weak_topic_match"] = df.apply(check_mapping, axis=1)

summary = (
    df.groupby(
        ["topic_label_ALL", "candidate_theme"],
        dropna=False
    )
    .agg(
        examples=("review_id", "size"),
        mapped_matches=("weak_topic_match", "sum"),
    )
    .reset_index()
)

summary["match_rate"] = (
    summary["mapped_matches"] / summary["examples"]
)

summary.to_csv(OUTPUT_PATH, index=False)

print("Mapped matches:", df["weak_topic_match"].sum())
print("Total examples:", len(df))
print(
    "Mapped match percentage:",
    round(df["weak_topic_match"].mean() * 100, 2),
    "%"
)

print("\nMapping audit:")
print(summary.to_string(index=False))

print("\nSaved:", OUTPUT_PATH)
