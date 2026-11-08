import pandas as pd
import pytest
from pathlib import Path


DASHBOARD_DIR = Path("data/processed/dashboard")


@pytest.fixture(scope="module")
def theme_summary():
    return pd.read_csv(DASHBOARD_DIR / "theme_summary.csv")


@pytest.fixture(scope="module")
def app_summary():
    return pd.read_csv(DASHBOARD_DIR / "app_summary.csv")


@pytest.fixture(scope="module")
def app_theme_summary():
    return pd.read_csv(DASHBOARD_DIR / "app_theme_summary.csv")


@pytest.fixture(scope="module")
def dashboard_reviews():
    return pd.read_parquet(
        DASHBOARD_DIR / "dashboard_reviews.parquet"
    )


def test_theme_summary_has_nine_themes(theme_summary):
    assert len(theme_summary) == 9
    assert theme_summary["theme"].nunique() == 9


def test_theme_summary_has_valid_counts(theme_summary):
    assert (theme_summary["reviews"] >= 0).all()
    assert (theme_summary["coverage_pct"] >= 0).all()
    assert (theme_summary["coverage_pct"] <= 100).all()


def test_app_summary_has_six_apps(app_summary):
    assert len(app_summary) == 6
    assert app_summary["app"].nunique() == 6


def test_app_summary_has_valid_metrics(app_summary):
    assert (app_summary["total_reviews"] > 0).all()
    assert app_summary["average_rating"].between(1, 5).all()


def test_app_theme_summary_has_six_apps(app_theme_summary):
    assert len(app_theme_summary) == 6
    assert app_theme_summary["app"].nunique() == 6


def test_app_theme_counts_are_non_negative(app_theme_summary):
    theme_columns = [
        column for column in app_theme_summary.columns
        if column.startswith("theme__")
    ]

    assert len(theme_columns) == 9
    assert (app_theme_summary[theme_columns] >= 0).all().all()


def test_dashboard_reviews_has_expected_columns(dashboard_reviews):
    expected_columns = {
        "review_id",
        "app",
        "score",
        "review_date",
        "review_text",
        "assigned_themes",
        "assigned_theme_count",
        "theme_status",
    }

    assert expected_columns.issubset(
        set(dashboard_reviews.columns)
    )


def test_dashboard_reviews_has_unique_ids(dashboard_reviews):
    assert dashboard_reviews["review_id"].is_unique


def test_dashboard_reviews_have_valid_scores(dashboard_reviews):
    assert dashboard_reviews["score"].between(1, 5).all()


def test_dashboard_review_theme_counts_are_consistent(
    dashboard_reviews
):
    actual_counts = dashboard_reviews[
        "assigned_themes"
    ].apply(len)

    assert (
        actual_counts ==
        dashboard_reviews["assigned_theme_count"]
    ).all()


def test_dashboard_reviews_have_valid_statuses(dashboard_reviews):
    valid_statuses = {
        "unassigned",
        "single_theme",
        "multiple_themes",
    }

    assert set(
        dashboard_reviews["theme_status"].unique()
    ).issubset(valid_statuses)


def test_dashboard_review_ids_match_final_dataset(
    dashboard_reviews
):
    final_data = pd.read_parquet(
        "data/processed/final_complaint_themes.parquet"
    )

    assert set(dashboard_reviews["review_id"]) == set(
        final_data["review_id"]
    )