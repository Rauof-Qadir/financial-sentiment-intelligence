import pandas as pd

INPUT_PATH = "data/processed/banking_complaints.parquet"

CORE_OUTPUT = "data/processed/banking_complaints_core.parquet"
BROAD_OUTPUT = "data/processed/banking_complaints_broad.parquet"

print("\nLoading complaint candidates...")

df = pd.read_parquet(INPUT_PATH)

# Classify feedback groups
df["complaint_cohort"] = "ambiguous_rating"

df.loc[
    df["score"].isin([1, 2]),
    "complaint_cohort"
] = "core_complaint"

df.loc[
    df["score"].isin([4, 5]),
    "complaint_cohort"
] = "mixed_or_conflicting_feedback"

# Core dataset: 1–2 stars
core = df[
    df["complaint_cohort"] == "core_complaint"
].copy()

# Broad dataset: all negative annotations
broad = df.copy()

# Save both datasets
core.to_parquet(CORE_OUTPUT, index=False)
broad.to_parquet(BROAD_OUTPUT, index=False)

print("\nCOHORT SUMMARY")
print("=" * 50)

print("\nAll candidates:", len(broad))
print("Core complaints:", len(core))

print("\nCohort distribution:")

print(
    broad["complaint_cohort"]
    .value_counts()
    .to_string()
)

print("\nCore complaint distribution by app:")

print(
    core["app"]
    .value_counts()
    .to_string()
)

print("\nSaved files:")
print(CORE_OUTPUT)
print(BROAD_OUTPUT)

print("\nCOMPLAINT COHORT CREATION COMPLETED.")