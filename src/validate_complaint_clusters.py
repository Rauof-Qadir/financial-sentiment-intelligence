
import re
import pandas as pd

INPUT_PATH = "data/processed/core_complaints_clustered_k10_v3.parquet"

OUTPUT_PATH = "reports/complaint_cluster_rule_validation.csv"
SUMMARY_PATH = "reports/complaint_cluster_rule_summary.csv"

# Each theme has keyword patterns.
THEME_RULES = {
    "Account restriction": [
        r"\bblocked account\b",
        r"\baccount blocked\b",
        r"\bfrozen account\b",
        r"\baccount frozen\b",
        r"\blocked account\b",
        r"\block(?:ed|ing)? my account\b",
        r"\bclosed account\b",
        r"\baccount closure\b",
        r"\brestricted account\b",
        r"\bcannot access (?:my )?account\b",
        r"\baccess to my funds\b",
    ],

    "Fraud/security": [
        r"\bfraud\b",
        r"\bscam(?:med|ming)?\b",
        r"\bstolen money\b",
        r"\bunauthori[sz]ed\b",
        r"\bidentity theft\b",
        r"\bsecurity breach\b",
        r"\bstole my money\b",
    ],

    "Money transfer": [
        r"\btransfer(?:s|red|ring)?\b",
        r"\bwithdraw(?:al|als)?\b",
        r"\bwithdraw money\b",
        r"\bsend money\b",
        r"\bdeposit money\b",
        r"\bmove money\b",
        r"\bfunds transfer\b",
        r"\bpayment failed\b",
    ],

    "Login/authentication": [
        r"\blog ?in\b",
        r"\bsign ?in\b",
        r"\bpasscode\b",
        r"\bpassword\b",
        r"\bfingerprint\b",
        r"\bverification code\b",
        r"\bactivation code\b",
        r"\bid verification\b",
        r"\blogged out\b",
        r"\bcannot access the app\b",
    ],

    "Customer support": [
        r"\bcustomer support\b",
        r"\bcustomer service\b",
        r"\blive chat\b",
        r"\bsupport team\b",
        r"\bhelp ?desk\b",
        r"\bno response\b",
        r"\bno reply\b",
        r"\bno one responds\b",
        r"\bcannot contact\b",
        r"\bphone support\b",
    ],

    "Card/payment": [
        r"\bcredit card\b",
        r"\bdebit card\b",
        r"\bcard payment\b",
        r"\bcard declined\b",
        r"\btop.?ups?\b",
        r"\bcontactless\b",
        r"\btransaction fee\b",
        r"\bexchange fee\b",
        r"\bsubscription\b",
        r"\bcharged\b",
        r"\bpayment\b",
    ],

    "App performance": [
        r"\bcrash(?:es|ed|ing)?\b",
        r"\bfreez(?:e|es|ing)\b",
        r"\bslow\b",
        r"\bnot responding\b",
        r"\bstopped working\b",
        r"\bdoesn't work\b",
        r"\bdoes not work\b",
        r"\bupdate\b",
        r"\bloading\b",
        r"\berror message\b",
        r"\bcompatib(?:le|ility)\b",
    ],
}

# Priority decides the primary theme when multiple themes match.
PRIORITY = [
    "Account restriction",
    "Fraud/security",
    "Money transfer",
    "Login/authentication",
    "Customer support",
    "Card/payment",
    "App performance",
]


def classify_review(text):
    text = str(text).lower()

    matched = []

    for theme, patterns in THEME_RULES.items():
        if any(re.search(pattern, text) for pattern in patterns):
            matched.append(theme)

    if not matched:
        return "Other/unclear", "", 0

    primary = next(
        theme for theme in PRIORITY if theme in matched
    )

    return primary, " | ".join(matched), len(matched)


def main():
    print("Loading clustered complaints...")

    df = pd.read_parquet(INPUT_PATH)

    print("Total complaints:", len(df))

    results = df["cleaned_text"].apply(classify_review)

    df["primary_theme"] = results.apply(lambda x: x[0])
    df["matched_themes"] = results.apply(lambda x: x[1])
    df["matched_theme_count"] = results.apply(lambda x: x[2])

    df["multiple_themes"] = df["matched_theme_count"] > 1
    df["unmatched"] = df["matched_theme_count"] == 0

    # Save review-level classifications.
    df.to_csv(OUTPUT_PATH, index=False)

    # Cluster-level summary.
    summary = df.groupby("cluster_id").agg(
        total_reviews=("cleaned_text", "size"),
        unmatched_reviews=("unmatched", "sum"),
        multi_theme_reviews=("multiple_themes", "sum"),
    )

    theme_counts = pd.crosstab(
        df["cluster_id"],
        df["primary_theme"]
    )

    summary = summary.join(theme_counts, how="left").fillna(0)

    summary["unmatched_pct"] = (
        summary["unmatched_reviews"] /
        summary["total_reviews"] * 100
    ).round(2)

    summary["multi_theme_pct"] = (
        summary["multi_theme_reviews"] /
        summary["total_reviews"] * 100
    ).round(2)

    summary.to_csv(SUMMARY_PATH)

    print("\nCLUSTER RULE VALIDATION")
    print("=" * 65)
    print(summary.to_string())

    print("\nOverall classification:")
    print(df["primary_theme"].value_counts().to_string())

    print("\nUnmatched reviews:", int(df["unmatched"].sum()))
    print("Multiple-theme reviews:", int(df["multiple_themes"].sum()))

    print("\nSaved reports:")
    print(OUTPUT_PATH)
    print(SUMMARY_PATH)


if __name__ == "__main__":
    main()
