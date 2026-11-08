import pandas as pd
import pytest
import numpy as np

DATA_PATH = "data/processed/final_complaint_themes.parquet"

VALID_THEMES = {
    "App_performance",
    "Login_authentication",
    "Account_restriction",
    "Fraud_security",
    "Money_transfer",
    "Customer_support",
    "Card_payment",
    "Features_usability",
    "Trading_investment",
}

THEME_COLUMNS = [
    f"theme__{theme}" for theme in VALID_THEMES
]


@pytest.fixture(scope="module")
def complaints():
    return pd.read_parquet(DATA_PATH)


def test_dataset_is_not_empty(complaints):
    assert len(complaints) > 0


def test_review_ids_are_unique(complaints):
    assert complaints["review_id"].is_unique


def test_all_theme_flags_are_boolean(complaints):
    assert all(
        complaints[column].dtype == bool
        for column in THEME_COLUMNS
    )


def test_assigned_themes_are_valid(complaints):
    for themes in complaints["assigned_themes"]:
        assert isinstance(
            themes,
            (list, tuple, np.ndarray)
        )

        assert set(themes).issubset(VALID_THEMES)


def test_theme_count_matches_assigned_list(complaints):
    actual_counts = complaints["assigned_themes"].apply(len)

    assert (
        actual_counts == complaints["assigned_theme_count"]
    ).all()


def test_theme_count_matches_boolean_flags(complaints):
    calculated_counts = complaints[THEME_COLUMNS].sum(axis=1)

    assert (
        calculated_counts == complaints["assigned_theme_count"]
    ).all()


def test_unassigned_status_is_consistent(complaints):
    mask = complaints["assigned_theme_count"] == 0

    assert (
        complaints.loc[mask, "theme_status"] == "unassigned"
    ).all()


def test_single_theme_status_is_consistent(complaints):
    mask = complaints["assigned_theme_count"] == 1

    assert (
        complaints.loc[mask, "theme_status"] == "single_theme"
    ).all()


def test_multiple_theme_status_is_consistent(complaints):
    mask = complaints["assigned_theme_count"] > 1

    assert (
        complaints.loc[mask, "theme_status"] == "multiple_themes"
    ).all()


def test_theme_status_has_no_unknown_values(complaints):
    valid_statuses = {
        "unassigned",
        "single_theme",
        "multiple_themes",
    }

    assert set(complaints["theme_status"].unique()).issubset(
        valid_statuses
    )


def test_theme_flags_match_assigned_list(complaints):
    for _, row in complaints.iterrows():
        assigned = set(row["assigned_themes"])

        for theme in VALID_THEMES:
            expected = theme in assigned
            actual = row[f"theme__{theme}"]

            assert actual == expected