
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

THEMES_V2 = {
    "App_performance":
        "Technical problems with the banking app itself: "
        "app crashes, freezes, fails to open, becomes unresponsive, "
        "loads slowly, or displays technical errors.",

    "Login_authentication":
        "Problems accessing an existing banking account: "
        "incorrect password, login failure, OTP not received, "
        "authentication failure, verification code, or sign-in problems.",

    "Account_restriction":
        "The bank restricts the customer's bank account or access "
        "to funds: account blocked, frozen, suspended, closed, "
        "limited, or unexpectedly locked by the bank.",

    "Fraud_security":
        "Financial crime or unauthorized activity: fraudulent "
        "transactions, scams, stolen credentials, identity theft, "
        "account hacking, or suspicious payments not authorized "
        "by the customer.",

    "Money_transfer":
        "Problems moving money between accounts or people: "
        "bank transfer failed, money not received, delayed transfer, "
        "incorrect recipient, sending money, receiving money, "
        "or withdrawing funds.",

    "Customer_support":
        "Problems getting help from the bank: unavailable support, "
        "unhelpful agents, unanswered messages, long response times, "
        "unresolved complaints, or poor customer service.",

    "Card_payment":
        "Problems specifically involving payment cards or card purchases: "
        "debit card declined, credit card declined, card activation, "
        "card frozen, card not working, or merchant card payment failure.",

    "Features_usability":
        "Problems understanding or using banking app features: "
        "confusing interface, difficult navigation, poor design, "
        "missing features, inconvenient workflows, or difficult settings.",

    "Trading_investment":
        "Problems or feedback about investment and trading services: "
        "buying or selling stocks, cryptocurrency trading, investment "
        "portfolios, trading fees, or investment account features."
}

model = SentenceTransformer(MODEL_NAME)

names = list(THEMES_V2.keys())
descriptions = list(THEMES_V2.values())

embeddings = model.encode(
    descriptions,
    normalize_embeddings=True,
    show_progress_bar=False
)

matrix = cosine_similarity(embeddings)

result = pd.DataFrame(
    matrix,
    index=names,
    columns=names
)

result.to_csv(
    "reports/theme_description_similarity_matrix_v2.csv"
)

print("THEME DESCRIPTION SIMILARITY V2")
print("=" * 70)

for theme in names:
    similarities = result.loc[theme].drop(theme)
    closest = similarities.idxmax()

    print(
        f"{theme:25s} | "
        f"Closest: {closest:25s} | "
        f"Similarity: {similarities.max():.4f}"
    )

print("\nSaved:")
print("reports/theme_description_similarity_matrix_v2.csv")
