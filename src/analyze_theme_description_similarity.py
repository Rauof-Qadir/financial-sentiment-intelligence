
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

THEMES = {
    "App_performance":
        "The mobile banking application crashes, freezes, "
        "loads slowly, stops working, or has technical errors.",

    "Login_authentication":
        "The customer cannot log in, authenticate, "
        "receive OTP codes, or access their account.",

    "Account_restriction":
        "The bank blocks, freezes, suspends, restricts, "
        "or closes the customer's account or funds.",

    "Fraud_security":
        "The customer reports fraud, scams, unauthorized "
        "transactions, suspicious activity, or security breaches.",

    "Money_transfer":
        "The customer has problems sending, receiving, "
        "transferring, or withdrawing money.",

    "Customer_support":
        "The customer complains about customer service, "
        "support agents, slow responses, or unresolved complaints.",

    "Card_payment":
        "The customer reports problems with debit cards, "
        "credit cards, card transactions, or payment processing.",

    "Features_usability":
        "The customer finds application features difficult "
        "to use, confusing, inconvenient, or poorly designed.",

    "Trading_investment":
        "The customer discusses trading, stocks, crypto, "
        "investment features, or investment account problems."
}

model = SentenceTransformer(MODEL_NAME)

names = list(THEMES.keys())
descriptions = list(THEMES.values())

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
    "reports/theme_description_similarity_matrix.csv"
)

print("THEME DESCRIPTION SIMILARITY")
print("=" * 70)

for i, theme in enumerate(names):
    similarities = result.loc[theme].drop(theme)
    closest = similarities.idxmax()
    score = similarities.max()

    print(
        f"{theme:25s} | "
        f"Closest: {closest:25s} | "
        f"Similarity: {score:.4f}"
    )

print("\nSaved:")
print("reports/theme_description_similarity_matrix.csv")
