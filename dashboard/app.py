from pathlib import Path
import json

import pandas as pd
import streamlit as st


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Financial Sentiment Intelligence",
    page_icon="📊",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "dashboard"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    theme_summary = pd.read_csv(
        DATA_DIR / "theme_summary.csv"
    )

    app_summary = pd.read_csv(
        DATA_DIR / "app_summary.csv"
    )

    app_theme = pd.read_csv(
        DATA_DIR / "app_theme_summary.csv"
    )

    reviews = pd.read_parquet(
        DATA_DIR / "dashboard_reviews.parquet"
    )

    reviews["review_date"] = pd.to_datetime(
        reviews["review_date"],
        errors="coerce"
    )

    news_predictions = pd.read_csv(
        PROJECT_ROOT / "reports" / "financial_news_test_predictions.csv"
    )

    with open(
        PROJECT_ROOT / "reports" / "financial_news_final_test_metrics.json",
        "r"
    ) as file:
        news_metrics = json.load(file)

    return (
        theme_summary,
        app_summary,
        app_theme,
        reviews,
        news_predictions,
        news_metrics
    )

try:
    (
    theme_summary,
    app_summary,
    app_theme,
    reviews,
    news_predictions,
    news_metrics
     ) = load_data()

except Exception as error:
    st.error(f"Could not load dashboard data: {error}")
    st.stop()


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📊 Financial Sentiment & Complaint Intelligence")

st.markdown(
    """
    Explore customer complaint patterns across six banking apps.
    Identify recurring issues, compare apps, and inspect individual reviews.
    """
)

st.caption(
    "Dataset: Banking app review complaint candidates | "
    "Theme assignments are provisional and not independently accuracy-validated."
)


# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.header("Dashboard Filters")

selected_apps = st.sidebar.multiselect(
    "Select banking apps",
    options=sorted(reviews["app"].dropna().unique()),
    default=sorted(reviews["app"].dropna().unique())
)

theme_options = theme_summary["theme"].tolist()

selected_theme = st.sidebar.selectbox(
    "Complaint theme",
    options=["All themes"] + theme_options
)

selected_rating = st.sidebar.multiselect(
    "Star rating",
    options=sorted(reviews["score"].dropna().unique()),
    default=sorted(reviews["score"].dropna().unique())
)

search_text = st.sidebar.text_input(
    "Search review text",
    placeholder="e.g. login, transfer, card..."
)


# --------------------------------------------------
# FILTER DATA
# --------------------------------------------------

filtered = reviews[
    reviews["app"].isin(selected_apps)
    & reviews["score"].isin(selected_rating)
].copy()

if selected_theme != "All themes":
    theme_key = selected_theme.lower().replace(" ", "_")

    filtered = filtered[
        filtered["assigned_themes"].apply(
            lambda themes: theme_key in themes
            if isinstance(themes, (list, tuple))
            else False
        )
    ]

if search_text.strip():
    filtered = filtered[
        filtered["review_text"]
        .fillna("")
        .str.contains(
            search_text.strip(),
            case=False,
            regex=False
        )
    ]


# --------------------------------------------------
# KPI CARDS
# --------------------------------------------------

total_reviews = len(filtered)

average_rating = (
    filtered["score"].mean()
    if total_reviews > 0
    else 0
)

multiple_theme_reviews = (
    (filtered["assigned_theme_count"] > 1).sum()
    if total_reviews > 0
    else 0
)

unassigned_reviews = (
    (filtered["theme_status"] == "unassigned").sum()
    if total_reviews > 0
    else 0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Complaint Reviews",
    f"{total_reviews:,}"
)

col2.metric(
    "Average Star Rating",
    f"{average_rating:.2f}" if total_reviews else "N/A"
)

col3.metric(
    "Multiple Themes",
    f"{multiple_theme_reviews:,}"
)

col4.metric(
    "Unassigned Reviews",
    f"{unassigned_reviews:,}"
)


# --------------------------------------------------
# OVERVIEW TABS
# --------------------------------------------------

overview_tab, explorer_tab, methodology_tab, news_tab = st.tabs(
    [
        "📈 Banking Overview",
        "🔎 Complaint Explorer",
        "ℹ️ Methodology",
        "📰 Financial News"
    ]
)


# --------------------------------------------------
# OVERVIEW
# --------------------------------------------------

with overview_tab:

    st.subheader("Complaint Theme Distribution")

    if selected_theme == "All themes":

        theme_chart = theme_summary.set_index("theme")[
            ["reviews"]
        ]

        st.bar_chart(
            theme_chart,
            horizontal=True
        )

        st.caption(
            "Theme counts overlap because one review can belong to multiple themes."
        )

    else:
        st.info(
            f"Showing reviews assigned to: {selected_theme}"
        )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Complaint Candidates by App")

        app_counts = (
            filtered["app"]
            .value_counts()
            .rename("reviews")
            .to_frame()
        )

        st.bar_chart(
            app_counts,
            horizontal=True
        )

    with right:
        st.subheader("App-wise Average Rating")

        rating_chart = (
            filtered.groupby("app")["score"]
            .mean()
            .sort_values()
            .to_frame("average_rating")
        )

        st.bar_chart(
            rating_chart,
            horizontal=True
        )

    st.divider()

    st.subheader("App vs Complaint Theme")

    theme_columns = [
        col for col in app_theme.columns
        if col != "app"
    ]

    app_theme_filtered = app_theme[
        app_theme["app"].isin(selected_apps)
    ].set_index("app")

    if selected_theme != "All themes":
        selected_column = (
            "theme__" + selected_theme.lower().replace(" ", "_")
        )

        if selected_column in app_theme_filtered.columns:
            app_theme_filtered = app_theme_filtered[
                [selected_column]
            ]

    else:
        app_theme_filtered = app_theme_filtered[theme_columns]

    st.dataframe(
        app_theme_filtered,
        use_container_width=True
    )


# --------------------------------------------------
# COMPLAINT EXPLORER
# --------------------------------------------------

with explorer_tab:

    st.subheader("Customer Complaint Explorer")

    st.write(
        f"Showing **{len(filtered):,}** matching reviews."
    )

    display_columns = [
        "app",
        "score",
        "review_date",
        "review_text",
        "assigned_themes",
        "theme_status"
    ]

    display_data = filtered[
        display_columns
    ].sort_values(
        "review_date",
        ascending=False,
        na_position="last"
    )

    st.dataframe(
        display_data,
        use_container_width=True,
        height=550
    )

    csv_data = display_data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download filtered reviews (CSV)",
        data=csv_data,
        file_name="filtered_customer_complaints.csv",
        mime="text/csv"
    )


# --------------------------------------------------
# METHODOLOGY
# --------------------------------------------------

with methodology_tab:

    st.subheader("How to interpret this dashboard")

    st.markdown(
        """
        **1. Complaint candidates**

        Reviews were selected using the dataset's existing negative
        sentiment annotation and usable review text. They are candidates,
        not individually verified complaints.

        **2. Theme classification**

        Sentence embeddings were compared with descriptions of nine
        complaint themes. A provisional cosine-similarity threshold
        of 0.40 was used for theme assignment.

        **3. Multiple themes**

        A review can be assigned more than one theme. Therefore,
        theme counts and coverage percentages can overlap.

        **4. Unassigned reviews**

        These reviews did not meet the selected similarity threshold
        for any theme. It does not mean they contain no useful feedback.

        **5. Rating interpretation**

        Star ratings describe customer satisfaction signals. They are
        not the same as independently verified text sentiment.

        **6. Limitations**

        Theme assignments have not been validated against a sufficiently
        large human-labelled test set. Treat the results as exploratory
        business intelligence, not ground-truth classification.

        **7. Financial interpretation**

        Customer complaint sentiment does not predict stock prices,
        investment returns, or financial-market direction.
        """
    )

st.divider()

st.caption(
    "Financial Sentiment Intelligence | Customer Feedback Analytics"
)

# --------------------------------------------------
# FINANCIAL NEWS INTELLIGENCE
# --------------------------------------------------

with news_tab:

    st.subheader("Financial News Sentiment Analysis")

    st.write(
        "Sentiment classification results on the held-out "
        "Financial PhraseBank test dataset."
    )

    # Model metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Test Accuracy",
        f"{news_metrics['accuracy']:.2%}"
    )

    col2.metric(
        "Macro Precision",
        f"{news_metrics['macro_precision']:.2%}"
    )

    col3.metric(
        "Macro Recall",
        f"{news_metrics['macro_recall']:.2%}"
    )

    col4.metric(
        "Macro F1",
        f"{news_metrics['macro_f1']:.2%}"
    )

    st.caption(
        f"Model: {news_metrics['model']} | "
        f"C: {news_metrics['C']}"
    )

    st.divider()

    # Sentiment distribution
    left, right = st.columns(2)

    sentiment_order = [
        "negative",
        "neutral",
        "positive"
    ]

    with left:

        st.subheader("Actual Sentiment Distribution")

        actual_counts = (
            news_predictions["actual_sentiment"]
            .value_counts()
            .reindex(sentiment_order, fill_value=0)
            .rename("articles")
            .to_frame()
        )

        st.bar_chart(
            actual_counts,
            horizontal=True
        )

    with right:

        st.subheader("Predicted Sentiment Distribution")

        predicted_counts = (
            news_predictions["predicted_sentiment"]
            .value_counts()
            .reindex(sentiment_order, fill_value=0)
            .rename("articles")
            .to_frame()
        )

        st.bar_chart(
            predicted_counts,
            horizontal=True
        )

    st.divider()

    # Confusion matrix
    st.subheader("Model Confusion Matrix")

    confusion = pd.crosstab(
        news_predictions["actual_sentiment"],
        news_predictions["predicted_sentiment"]
    )

    confusion = confusion.reindex(
        index=sentiment_order,
        columns=sentiment_order,
        fill_value=0
    )

    st.dataframe(
        confusion,
        use_container_width=True
    )

    st.caption(
        "Rows represent actual sentiment; columns represent predicted sentiment. "
        "Diagonal values are correct predictions."
    )

    st.divider()

    # Prediction explorer
    st.subheader("Financial News Explorer")

    selected_sentiment = st.selectbox(
        "Filter by predicted sentiment",
        ["All", "negative", "neutral", "positive"],
        key="news_sentiment_filter"
    )

    news_search = st.text_input(
        "Search financial news",
        placeholder="Search company, event, or news text...",
        key="news_search"
    )

    filtered_news = news_predictions.copy()

    if selected_sentiment != "All":

        filtered_news = filtered_news[
            filtered_news["predicted_sentiment"]
            == selected_sentiment
        ]

    if news_search.strip():

        filtered_news = filtered_news[
            filtered_news["sentence"]
            .fillna("")
            .str.contains(
                news_search.strip(),
                case=False,
                regex=False
            )
        ]

    st.write(
        f"Matching articles: {len(filtered_news):,}"
    )

    news_display = filtered_news[
        [
            "sentence",
            "actual_sentiment",
            "predicted_sentiment",
            "is_correct"
        ]
    ]

    st.dataframe(
        news_display,
        use_container_width=True,
        height=450
    )