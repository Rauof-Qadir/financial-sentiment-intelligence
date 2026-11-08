from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Financial Sentiment Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "dashboard"
REPORT_DIR = PROJECT_ROOT / "reports"

THEME_COLORS = [
    "#6C8CFF", "#26C6DA", "#A78BFA", "#F59E0B", "#34D399",
    "#F472B6", "#FB7185", "#60A5FA", "#A3E635",
]
SENTIMENT_COLORS = {
    "negative": "#F87171",
    "neutral": "#FBBF24",
    "positive": "#34D399",
}
NAVY = "#101827"


# ============================================================
# PREMIUM VISUAL SYSTEM
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --bg: #0b1120;
        --panel: #111c2e;
        --panel-2: #152238;
        --border: rgba(148, 163, 184, .16);
        --muted: #91a0b8;
        --text: #edf3ff;
        --blue: #7c9cff;
        --cyan: #49d5e8;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
          radial-gradient(ellipse at 12% 0%, rgba(63, 94, 251, .14), transparent 34%),
          radial-gradient(ellipse at 90% 8%, rgba(34, 211, 238, .08), transparent 30%),
          #0b1120;
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(11, 17, 32, .72);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101a2c 0%, #0d1626 100%);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: #dce7f8;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2.0rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -.035em;
        color: #f4f7ff !important;
    }

    h1 { font-size: clamp(2rem, 4vw, 3.15rem) !important; }
    h2 { font-size: 1.65rem !important; }
    h3 { font-size: 1.18rem !important; }

    p, li, label, .stMarkdown {
        color: #c4d0e2;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.2rem 2.4rem;
        border-radius: 24px;
        border: 1px solid rgba(145, 166, 255, .25);
        background:
          radial-gradient(circle at 85% 15%, rgba(73, 213, 232, .22), transparent 28%),
          linear-gradient(120deg, rgba(37, 57, 103, .96), rgba(17, 31, 53, .98) 58%, rgba(17, 42, 62, .96));
        box-shadow: 0 24px 70px rgba(0, 0, 0, .22);
        margin-bottom: 1.35rem;
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 250px;
        height: 250px;
        right: -85px;
        bottom: -155px;
        border: 1px solid rgba(255,255,255,.13);
        border-radius: 50%;
        box-shadow: 0 0 0 28px rgba(255,255,255,.025),
                    0 0 0 58px rgba(255,255,255,.02);
    }

    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: .45rem;
        padding: .36rem .7rem;
        border: 1px solid rgba(154, 183, 255, .28);
        border-radius: 999px;
        background: rgba(124, 156, 255, .12);
        color: #c8d7ff;
        font-size: .74rem;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
        margin-bottom: .9rem;
    }

    .hero-title {
        color: #fff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2rem, 4.3vw, 3.35rem);
        line-height: 1.08;
        font-weight: 700;
        max-width: 850px;
        letter-spacing: -.05em;
        margin: 0 0 .85rem 0;
    }

    .hero-copy {
        color: #c2d2ed;
        font-size: 1.02rem;
        line-height: 1.7;
        max-width: 760px;
        margin: 0;
    }

    .hero-meta {
        display: flex;
        flex-wrap: wrap;
        gap: .55rem;
        margin-top: 1.35rem;
    }

    .meta-pill {
        border: 1px solid rgba(203, 216, 255, .18);
        background: rgba(8, 17, 34, .25);
        color: #d7e4ff;
        border-radius: 999px;
        padding: .38rem .72rem;
        font-size: .76rem;
        font-weight: 600;
    }

    .section-kicker {
        color: #7f9cff;
        font-size: .72rem;
        font-weight: 800;
        letter-spacing: .15em;
        text-transform: uppercase;
        margin: .25rem 0 .3rem 0;
    }

    .section-subtitle {
        color: #91a0b8;
        font-size: .91rem;
        margin-top: -.4rem;
        margin-bottom: 1.1rem;
    }

    .metric-card {
        min-height: 138px;
        padding: 1.1rem 1.2rem;
        border-radius: 18px;
        border: 1px solid var(--border);
        background: linear-gradient(145deg, rgba(24, 39, 62, .96), rgba(15, 27, 45, .96));
        box-shadow: 0 12px 32px rgba(0,0,0,.12);
    }

    .metric-label {
        color: #9aabc4;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .06em;
        text-transform: uppercase;
    }

    .metric-value {
        color: #f5f8ff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -.04em;
        margin-top: .35rem;
        line-height: 1.15;
    }

    .metric-note {
        color: #8192ad;
        font-size: .76rem;
        margin-top: .45rem;
    }

    .panel {
        border: 1px solid var(--border);
        border-radius: 18px;
        background: linear-gradient(145deg, rgba(20, 34, 55, .92), rgba(14, 25, 42, .92));
        padding: 1.15rem 1.25rem;
    }

    .insight {
        border-left: 3px solid #6c8cff;
        padding: .85rem 1rem;
        background: rgba(108, 140, 255, .08);
        border-radius: 0 12px 12px 0;
        color: #cbd8ee;
        font-size: .88rem;
        line-height: 1.6;
    }

    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #34d399;
        margin-right: .45rem;
        box-shadow: 0 0 12px rgba(52, 211, 153, .7);
    }

    div[data-testid="stMetric"] {
        background: transparent;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: .45rem;
        background: rgba(17, 28, 46, .75);
        padding: .45rem;
        border: 1px solid var(--border);
        border-radius: 14px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 45px;
        border-radius: 10px;
        color: #aebbd0;
        font-weight: 700;
        padding: 0 1rem;
    }

    .stTabs [aria-selected="true"] {
        background: #263b64 !important;
        color: white !important;
    }

    .stButton button, .stDownloadButton button {
        border-radius: 10px;
        border: 1px solid rgba(124,156,255,.35);
        background: linear-gradient(110deg, #5478e8, #4966c9);
        color: white;
        font-weight: 700;
    }

    .stButton button:hover, .stDownloadButton button:hover {
        border-color: #9db3ff;
        color: white;
        background: linear-gradient(110deg, #6688f2, #5978df);
    }

    input, textarea, [data-baseweb="select"] > div {
        background-color: #142238 !important;
        border-color: rgba(148,163,184,.2) !important;
        border-radius: 10px !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    hr {
        border-color: rgba(148,163,184,.14);
    }

    .footer {
        color: #71829d;
        text-align: center;
        font-size: .78rem;
        padding: 1.8rem 0 .5rem 0;
    }

    @media (max-width: 700px) {
        .hero { padding: 1.5rem; }
        .metric-value { font-size: 1.6rem; }
        .block-container { padding-top: 1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING AND VALIDATION
# ============================================================

@st.cache_data(show_spinner="Loading intelligence datasets...")
def load_data():
    required_files = [
        DATA_DIR / "theme_summary.csv",
        DATA_DIR / "app_summary.csv",
        DATA_DIR / "app_theme_summary.csv",
        DATA_DIR / "dashboard_reviews.parquet",
        REPORT_DIR / "financial_news_test_predictions.csv",
        REPORT_DIR / "financial_news_final_test_metrics.json",
    ]

    missing = [path for path in required_files if not path.exists()]
    if missing:
        missing_text = "\n".join(f"- {path.relative_to(PROJECT_ROOT)}" for path in missing)
        raise FileNotFoundError(
            "Some dashboard artifacts are missing. Generate the processed datasets "
            "and financial-news evaluation reports first:\n" + missing_text
        )

    theme_summary = pd.read_csv(required_files[0])
    app_summary = pd.read_csv(required_files[1])
    app_theme = pd.read_csv(required_files[2])
    reviews = pd.read_parquet(required_files[3])
    news_predictions = pd.read_csv(required_files[4])

    with open(required_files[5], "r", encoding="utf-8") as file:
        news_metrics = json.load(file)

    reviews["review_date"] = pd.to_datetime(reviews["review_date"], errors="coerce")

    required_review_columns = {
        "app", "score", "review_date", "review_text",
        "assigned_themes", "assigned_theme_count", "theme_status",
    }
    missing_columns = required_review_columns - set(reviews.columns)
    if missing_columns:
        raise ValueError(f"Dashboard review data is missing columns: {sorted(missing_columns)}")

    required_news_columns = {
        "sentence", "actual_sentiment", "predicted_sentiment", "is_correct",
    }
    missing_news_columns = required_news_columns - set(news_predictions.columns)
    if missing_news_columns:
        raise ValueError(f"Financial news predictions are missing columns: {sorted(missing_news_columns)}")

    return theme_summary, app_summary, app_theme, reviews, news_predictions, news_metrics


try:
    (
        theme_summary,
        app_summary,
        app_theme,
        reviews,
        news_predictions,
        news_metrics,
    ) = load_data()
except Exception as error:
    st.error("The dashboard could not load its data artifacts.")
    st.code(str(error))
    st.info(
        "Run the project's data preparation, model evaluation, complaint-theme, "
        "and dashboard-data scripts before launching Streamlit."
    )
    st.stop()


# ============================================================
# HELPERS
# ============================================================

def fmt_number(value: int | float) -> str:
    return f"{value:,.0f}"


def safe_theme_list(value: Any) -> list:
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    return []


def metric_card(label: str, value: str, note: str, accent: str = "#7c9cff"):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{accent}">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_heading(kicker: str, title: str, subtitle: str | None = None):
    st.markdown(f'<div class="section-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def style_plot(fig, height: int = 360):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#c6d2e5", size=12),
        margin=dict(l=12, r=18, t=25, b=12),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c6d2e5"),
        ),
        xaxis=dict(
            gridcolor="rgba(148,163,184,.12)",
            zerolinecolor="rgba(148,163,184,.12)",
            linecolor="rgba(148,163,184,.12)",
        ),
        yaxis=dict(
            gridcolor="rgba(148,163,184,.12)",
            zerolinecolor="rgba(148,163,184,.12)",
            linecolor="rgba(148,163,184,.12)",
        ),
    )
    return fig


def theme_filter_match(themes: Any, theme_key: str) -> bool:
    return theme_key in safe_theme_list(themes)


def empty_state(message: str):
    st.markdown(
        f'<div class="panel" style="text-align:center;color:#9aabc4;padding:2rem">{message}</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR AND FILTERS
# ============================================================

with st.sidebar:
    st.markdown("## 📊 FSI")
    st.caption("FINANCIAL INTELLIGENCE SUITE")
    st.markdown("---")

    st.markdown("### Workspace")
    selected_apps = st.multiselect(
        "Banking applications",
        options=sorted(reviews["app"].dropna().unique()),
        default=sorted(reviews["app"].dropna().unique()),
    )

    theme_options = theme_summary["theme"].dropna().tolist()
    selected_theme = st.selectbox(
        "Complaint theme",
        options=["All themes"] + theme_options,
    )

    selected_rating = st.multiselect(
        "Star rating",
        options=sorted(reviews["score"].dropna().unique()),
        default=sorted(reviews["score"].dropna().unique()),
    )

    search_text = st.text_input(
        "Search customer reviews",
        placeholder="Try: login, transfer, card...",
    )

    st.markdown("---")
    st.markdown("### Dataset snapshot")
    st.markdown(f"**{len(reviews):,}** complaint-candidate reviews")
    st.markdown(f"**{reviews['app'].nunique()}** banking applications")
    st.markdown(f"**{len(theme_options)}** semantic themes")
    st.markdown("---")
    st.caption("Exploratory analytics • Not investment advice")


filtered = reviews[
    reviews["app"].isin(selected_apps)
    & reviews["score"].isin(selected_rating)
].copy()

if selected_theme != "All themes":
    theme_key = selected_theme.lower().replace(" ", "_")
    filtered = filtered[
        filtered["assigned_themes"].apply(lambda themes: theme_filter_match(themes, theme_key))
    ]

if search_text.strip():
    filtered = filtered[
        filtered["review_text"].fillna("").str.contains(
            search_text.strip(), case=False, regex=False
        )
    ]

total_reviews = len(filtered)
average_rating = filtered["score"].mean() if total_reviews else 0
multiple_theme_reviews = (
    int((filtered["assigned_theme_count"] > 1).sum()) if total_reviews else 0
)
unassigned_reviews = (
    int((filtered["theme_status"] == "unassigned").sum()) if total_reviews else 0
)
multi_theme_pct = multiple_theme_reviews / total_reviews if total_reviews else 0


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">✦ NLP · MACHINE LEARNING · FINANCIAL ANALYTICS</div>
        <div class="hero-title">Financial Sentiment<br>Intelligence</div>
        <p class="hero-copy">
            Turn financial text into actionable analytical insight.
            Explore banking customer experience, uncover recurring complaint
            themes, and evaluate sentiment in financial news — all in one workspace.
        </p>
        <div class="hero-meta">
            <span class="meta-pill">Banking Customer Intelligence</span>
            <span class="meta-pill">Semantic Complaint Discovery</span>
            <span class="meta-pill">Financial News NLP</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<span class="status-dot"></span><span style="color:#8fa2bf;font-size:.8rem">DATA WORKSPACE READY</span>',
    unsafe_allow_html=True,
)

# KPI row
k1, k2, k3, k4 = st.columns(4)
with k1:
    metric_card("Matching reviews", fmt_number(total_reviews), "Current filter selection", "#8ea8ff")
with k2:
    metric_card("Average rating", f"{average_rating:.2f}/5" if total_reviews else "—", "Star-rating satisfaction signal", "#fbbf24")
with k3:
    metric_card("Multiple themes", fmt_number(multiple_theme_reviews), f"{multi_theme_pct:.1%} of matching reviews", "#42d6c5")
with k4:
    metric_card("Unassigned", fmt_number(unassigned_reviews), "No theme met the similarity threshold", "#c4a2ff")

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)


# ============================================================
# MAIN NAVIGATION
# ============================================================

overview_tab, explorer_tab, news_tab, methodology_tab = st.tabs(
    ["◈  Banking Overview", "⌕  Complaint Explorer", "▤  Financial News", "◎  Methodology"]
)


# ============================================================
# BANKING OVERVIEW
# ============================================================

with overview_tab:
    section_heading(
        "CUSTOMER EXPERIENCE INTELLIGENCE",
        "Banking overview",
        "A portfolio-level view of customer complaint patterns and app-level experience signals.",
    )

    left, right = st.columns([1.35, 1])

    with left:
        st.markdown("### Complaint theme landscape")
        if selected_theme == "All themes":
            chart_data = theme_summary.sort_values("reviews", ascending=True)
            fig = px.bar(
                chart_data,
                x="reviews",
                y="theme",
                orientation="h",
                text="reviews",
                color="theme",
                color_discrete_sequence=THEME_COLORS,
            )
            fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False)
            fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="Reviews")
            st.plotly_chart(style_plot(fig, max(360, 42 * len(chart_data))), use_container_width=True)
            st.caption("Theme assignments overlap: a single review can contribute to multiple themes.")
        else:
            st.markdown(
                f'<div class="insight">The workspace is filtered to <b>{selected_theme}</b>. '
                "Use the complaint explorer to inspect the individual reviews assigned to this theme.</div>",
                unsafe_allow_html=True,
            )
            theme_row = theme_summary[theme_summary["theme"] == selected_theme]
            if not theme_row.empty:
                st.metric("Dataset-wide reviews assigned to theme", fmt_number(theme_row.iloc[0]["reviews"]))

    with right:
        st.markdown("### Theme coverage")
        coverage = theme_summary.sort_values("coverage_pct", ascending=True)
        fig = px.bar(
            coverage,
            x="coverage_pct",
            y="theme",
            orientation="h",
            color="coverage_pct",
            color_continuous_scale=["#263b64", "#6c8cff", "#49d5e8"],
            text="coverage_pct",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
        fig.update_layout(
            coloraxis_showscale=False,
            yaxis_title="",
            xaxis_title="Coverage (%)",
        )
        st.plotly_chart(style_plot(fig, max(360, 42 * len(coverage))), use_container_width=True)
        st.caption("Coverage is based on the complaint review cohort; themes can overlap.")

    st.markdown("---")
    section_heading("APP COMPARISON", "Where are customers reporting issues?", "Compare complaint-candidate volume and average star ratings across selected apps.")

    app_left, app_right = st.columns(2)

    with app_left:
        app_counts = (
            filtered["app"].value_counts().rename_axis("app").reset_index(name="reviews")
        )
        if not app_counts.empty:
            fig = px.bar(
                app_counts.sort_values("reviews"),
                x="reviews",
                y="app",
                orientation="h",
                text="reviews",
                color="reviews",
                color_continuous_scale=["#293d68", "#6c8cff", "#49d5e8"],
            )
            fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False)
            fig.update_layout(coloraxis_showscale=False, xaxis_title="Matching reviews", yaxis_title="")
            st.plotly_chart(style_plot(fig, 320), use_container_width=True)
        else:
            empty_state("No reviews match the current filters.")

    with app_right:
        rating_by_app = (
            filtered.groupby("app", as_index=False)["score"]
            .mean()
            .sort_values("score")
        )
        if not rating_by_app.empty:
            fig = px.bar(
                rating_by_app,
                x="score",
                y="app",
                orientation="h",
                text="score",
                color="score",
                color_continuous_scale=["#F87171", "#FBBF24", "#34D399"],
                range_color=(1, 5),
            )
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
            fig.update_layout(
                coloraxis_showscale=False,
                xaxis_title="Average star rating",
                yaxis_title="",
                xaxis=dict(range=[0, 5.5]),
            )
            st.plotly_chart(style_plot(fig, 320), use_container_width=True)
        else:
            empty_state("No rating data available for the current filters.")

    st.markdown("---")
    section_heading("CROSS-APP ANALYSIS", "Complaint theme matrix", "Counts of theme assignments by banking application.")

    theme_columns = [column for column in app_theme.columns if column != "app"]
    app_theme_filtered = app_theme[app_theme["app"].isin(selected_apps)].copy()
    if selected_theme != "All themes":
        selected_column = "theme__" + selected_theme.lower().replace(" ", "_")
        if selected_column in app_theme_filtered.columns:
            app_theme_filtered = app_theme_filtered[["app", selected_column]]
    else:
        app_theme_filtered = app_theme_filtered[["app"] + theme_columns]

    if not app_theme_filtered.empty:
        st.dataframe(
            app_theme_filtered.set_index("app"),
            use_container_width=True,
            height=250,
        )
    else:
        empty_state("No app-level records match the selected apps.")

    st.markdown(
        '<div class="insight">Interpretation note: complaint volume is influenced by the number of available reviews '
        "and the dataset's collection process. It should not be treated as a market-share-adjusted complaint rate.</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# COMPLAINT EXPLORER
# ============================================================

with explorer_tab:
    section_heading(
        "REVIEW-LEVEL INVESTIGATION",
        "Complaint explorer",
        "Search, filter, inspect, and export the customer reviews behind the aggregate patterns.",
    )

    count_col, date_col, status_col = st.columns(3)
    with count_col:
        st.markdown(f"**{len(filtered):,}** matching records")
    with date_col:
        if filtered["review_date"].notna().any():
            st.markdown(f"**{filtered['review_date'].min():%d %b %Y} – {filtered['review_date'].max():%d %b %Y}**")
        else:
            st.markdown("**Date range unavailable**")
    with status_col:
        if len(filtered):
            st.markdown(f"**{(filtered['theme_status'] == 'unassigned').mean():.1%}** unassigned")
        else:
            st.markdown("**—** unassigned")

    st.markdown("---")

    display_columns = [
        "app", "score", "review_date", "review_text",
        "assigned_themes", "assigned_theme_count", "theme_status",
    ]
    display_data = filtered[display_columns].copy()
    display_data["assigned_themes"] = display_data["assigned_themes"].apply(
        lambda value: ", ".join(map(str, safe_theme_list(value)))
    )
    display_data = display_data.sort_values(
        "review_date", ascending=False, na_position="last"
    )

    if display_data.empty:
        empty_state("No reviews found. Try changing the sidebar filters or search term.")
    else:
        st.dataframe(
            display_data,
            use_container_width=True,
            height=520,
            column_config={
                "app": st.column_config.TextColumn("Banking app", width="medium"),
                "score": st.column_config.NumberColumn("Stars", format="%d ⭐", width="small"),
                "review_date": st.column_config.DateColumn("Review date", format="DD MMM YYYY"),
                "review_text": st.column_config.TextColumn("Customer review", width="large"),
                "assigned_themes": st.column_config.TextColumn("Assigned themes", width="large"),
                "assigned_theme_count": st.column_config.NumberColumn("Theme count", width="small"),
                "theme_status": st.column_config.TextColumn("Assignment status", width="medium"),
            },
        )

        csv_data = display_data.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "↓  Download filtered reviews (CSV)",
            data=csv_data,
            file_name="financial_sentiment_filtered_reviews.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.markdown("### Theme assignment profile")
    if not filtered.empty:
        status_counts = (
            filtered["theme_status"].value_counts().rename_axis("status").reset_index(name="reviews")
        )
        fig = px.pie(
            status_counts,
            names="status",
            values="reviews",
            hole=.68,
            color="status",
            color_discrete_map={
                "unassigned": "#64748b",
                "single": "#6c8cff",
                "multiple": "#49d5e8",
            },
        )
        fig.update_traces(textposition="outside", textinfo="label+percent")
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_plot(fig, 330), use_container_width=True)
    else:
        empty_state("Assignment profile will appear when matching reviews are available.")


# ============================================================
# FINANCIAL NEWS INTELLIGENCE
# ============================================================

with news_tab:
    section_heading(
        "FINANCIAL LANGUAGE ANALYTICS",
        "Financial news intelligence",
        "Held-out Financial PhraseBank test results and sentence-level model predictions.",
    )

    accuracy = float(news_metrics.get("accuracy", 0))
    macro_precision = float(news_metrics.get("macro_precision", 0))
    macro_recall = float(news_metrics.get("macro_recall", 0))
    macro_f1 = float(news_metrics.get("macro_f1", 0))
    weighted_f1 = float(news_metrics.get("weighted_f1", 0))

    n1, n2, n3, n4 = st.columns(4)
    with n1:
        metric_card("Test accuracy", f"{accuracy:.1%}", "Held-out test set", "#8ea8ff")
    with n2:
        metric_card("Macro precision", f"{macro_precision:.1%}", "Equal class weighting", "#49d5e8")
    with n3:
        metric_card("Macro recall", f"{macro_recall:.1%}", "Equal class weighting", "#fbbf24")
    with n4:
        metric_card("Macro F1", f"{macro_f1:.1%}", "Primary balanced metric", "#c4a2ff")

    st.markdown(
        f'<div class="insight">Selected model: <b>{news_metrics.get("model", "Not specified")}</b>'
        f' &nbsp;·&nbsp; C = <b>{news_metrics.get("C", "N/A")}</b>'
        f' &nbsp;·&nbsp; Weighted F1 = <b>{weighted_f1:.1%}</b>. '
        "These metrics describe classification performance on this dataset, not financial market prediction.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    left, right = st.columns(2)

    sentiment_order = ["negative", "neutral", "positive"]
    sentiment_palette = [SENTIMENT_COLORS[label] for label in sentiment_order]

    with left:
        st.markdown("### Actual sentiment distribution")
        actual_counts = (
            news_predictions["actual_sentiment"]
            .value_counts()
            .reindex(sentiment_order, fill_value=0)
            .rename_axis("sentiment")
            .reset_index(name="articles")
        )
        fig = px.bar(
            actual_counts,
            x="sentiment",
            y="articles",
            color="sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            text="articles",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Articles")
        st.plotly_chart(style_plot(fig, 320), use_container_width=True)

    with right:
        st.markdown("### Predicted sentiment distribution")
        predicted_counts = (
            news_predictions["predicted_sentiment"]
            .value_counts()
            .reindex(sentiment_order, fill_value=0)
            .rename_axis("sentiment")
            .reset_index(name="articles")
        )
        fig = px.bar(
            predicted_counts,
            x="sentiment",
            y="articles",
            color="sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            text="articles",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Articles")
        st.plotly_chart(style_plot(fig, 320), use_container_width=True)

    st.markdown("---")
    st.markdown("### Confusion matrix")

    confusion = pd.crosstab(
        news_predictions["actual_sentiment"],
        news_predictions["predicted_sentiment"],
    ).reindex(index=sentiment_order, columns=sentiment_order, fill_value=0)

    fig = go.Figure(
        data=go.Heatmap(
            z=confusion.values,
            x=sentiment_order,
            y=sentiment_order,
            colorscale=[
                [0, "#152238"],
                [.45, "#35528c"],
                [1, "#49d5e8"],
            ],
            text=confusion.values,
            texttemplate="%{text}",
            textfont=dict(size=17, color="#ffffff"),
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            colorbar=dict(title="Count"),
        )
    )
    fig.update_layout(
        xaxis_title="Predicted sentiment",
        yaxis_title="Actual sentiment",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(style_plot(fig, 390), use_container_width=True)
    st.caption("Rows represent actual sentiment; columns represent predicted sentiment. Diagonal cells are correct predictions.")

    st.markdown("---")
    st.markdown("### Financial news explorer")

    filter_col, search_col = st.columns([1, 2])
    with filter_col:
        selected_sentiment = st.selectbox(
            "Predicted sentiment",
            ["All", "negative", "neutral", "positive"],
            key="news_sentiment_filter",
        )
    with search_col:
        news_search = st.text_input(
            "Search news text",
            placeholder="Search company, event, or statement...",
            key="news_search",
        )

    filtered_news = news_predictions.copy()
    if selected_sentiment != "All":
        filtered_news = filtered_news[
            filtered_news["predicted_sentiment"] == selected_sentiment
        ]
    if news_search.strip():
        filtered_news = filtered_news[
            filtered_news["sentence"].fillna("").str.contains(
                news_search.strip(), case=False, regex=False
            )
        ]

    st.caption(f"{len(filtered_news):,} matching news statements")

    news_display = filtered_news[
        ["sentence", "actual_sentiment", "predicted_sentiment", "is_correct"]
    ].copy()

    if not news_display.empty:
        st.dataframe(
            news_display,
            use_container_width=True,
            height=430,
            column_config={
                "sentence": st.column_config.TextColumn("Financial news statement", width="large"),
                "actual_sentiment": st.column_config.TextColumn("Actual"),
                "predicted_sentiment": st.column_config.TextColumn("Predicted"),
                "is_correct": st.column_config.CheckboxColumn("Correct"),
            },
        )
        st.download_button(
            "↓  Download filtered news predictions",
            data=news_display.to_csv(index=False).encode("utf-8-sig"),
            file_name="financial_news_predictions.csv",
            mime="text/csv",
        )
    else:
        empty_state("No news statements match this search.")


# ============================================================
# METHODOLOGY
# ============================================================

with methodology_tab:
    section_heading(
        "TRANSPARENCY & INTERPRETATION",
        "Methodology and responsible use",
        "Understand what the platform measures, how themes are assigned, and where the evidence has limits.",
    )

    st.markdown("### Analytical workflows")
    workflow_col1, workflow_col2 = st.columns(2)

    with workflow_col1:
        st.markdown(
            """
            <div class="panel">
                <div class="section-kicker">WORKFLOW 01</div>
                <h3>Banking customer intelligence</h3>
                <p>Review text preparation → sentiment annotation analysis → complaint candidate selection → semantic theme assignment → dashboard exploration.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with workflow_col2:
        st.markdown(
            """
            <div class="panel">
                <div class="section-kicker">WORKFLOW 02</div>
                <h3>Financial news intelligence</h3>
                <p>Financial PhraseBank splits → word and character TF-IDF → model comparison → held-out evaluation → prediction exploration.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### How to interpret the banking analysis")

    methodology_items = [
        (
            "01",
            "Complaint candidates",
            "Reviews were selected using the dataset's existing negative sentiment annotation and usable review text. They are candidates, not individually verified complaints.",
        ),
        (
            "02",
            "Theme classification",
            "Sentence embeddings were compared with descriptions of nine complaint themes. A provisional cosine-similarity threshold of 0.40 was used for theme assignment.",
        ),
        (
            "03",
            "Multiple themes",
            "A review can be assigned more than one theme. Therefore, theme counts and coverage percentages can overlap.",
        ),
        (
            "04",
            "Unassigned reviews",
            "These reviews did not meet the selected similarity threshold for any theme. This does not mean they contain no useful feedback.",
        ),
        (
            "05",
            "Rating interpretation",
            "Star ratings describe customer satisfaction signals. They are not the same as independently verified text sentiment.",
        ),
        (
            "06",
            "Validation limitations",
            "Theme assignments have not been validated against a sufficiently large human-labelled test set. Treat results as exploratory business intelligence, not ground-truth classification.",
        ),
        (
            "07",
            "Financial interpretation",
            "Customer complaint sentiment and financial news sentiment do not predict stock prices, investment returns, or financial-market direction.",
        ),
    ]

    for number, title, description in methodology_items:
        st.markdown(
            f"""
            <div class="panel" style="margin-bottom:.65rem;display:flex;gap:1rem;align-items:flex-start">
                <div style="font-family:'Space Grotesk';font-size:1.2rem;font-weight:700;color:#7c9cff;min-width:2.2rem">{number}</div>
                <div><div style="color:#f1f5ff;font-weight:700;margin-bottom:.25rem">{title}</div>
                <div style="color:#9eacc2;font-size:.9rem;line-height:1.6">{description}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### Dataset and model notes")
    st.markdown(
        """
        - Banking review sentiment uses the dataset's existing annotations; they are not independently verified human ground truth.
        - Rating-derived sentiment is a satisfaction proxy, not a verified text label.
        - Semantic similarity values are scores, not calibrated probabilities.
        - The financial news classifier is evaluated on a held-out Financial PhraseBank test split.
        - Financial news classification is not a trading signal or investment recommendation.
        """
    )

    st.markdown(
        '<div class="insight">Responsible interpretation: use these outputs to investigate patterns and generate hypotheses. '
        "Validate important decisions with human review, representative samples, and domain expertise.</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    """
    <div class="footer">
        <b style="color:#aebdd4">FINANCIAL SENTIMENT INTELLIGENCE</b><br>
        NLP-powered customer experience and financial text analytics<br>
        <span style="opacity:.75">Exploratory research platform · Not investment advice</span>
    </div>
    """,
    unsafe_allow_html=True,
)
