
from pathlib import Path
import pandas as pd

# Project's raw data directory
DATA_DIR = Path("data/raw")

# All four datasets
DATASETS = {
    "Banking App Reviews": "banking_app_reviews.parquet",
    "Financial News Train": "financial_news_train.parquet",
    "Financial News Validation": "financial_news_validation.parquet",
    "Financial News Test": "financial_news_test.parquet",
}


def inspect_dataset(name: str, filename: str) -> None:
    """Load and display a dataset's basic structure."""

    file_path = DATA_DIR / filename

    print("\n" + "=" * 75)
    print(f"DATASET: {name}")
    print("=" * 75)

    df = pd.read_parquet(file_path)

    print(f"\nRows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    print("\nCOLUMN NAMES:")
    print(df.columns.tolist())

    print("\nDATA TYPES:")
    print(df.dtypes)

    print("\nMISSING VALUES:")
    print(df.isnull().sum())

    print(f"\nDUPLICATE ROWS: {df.duplicated().sum():,}")

    print("\nFIRST 3 RECORDS:")
    print(df.head(3).to_string(index=False))

    print("\nNUMERICAL SUMMARY:")
    print(df.describe(include="all").transpose().to_string())


def main():
    for name, filename in DATASETS.items():
        inspect_dataset(name, filename)


if __name__ == "__main__":
    main()
