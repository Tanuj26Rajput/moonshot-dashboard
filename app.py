import math
import re
from collections import Counter
from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client_groq = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)


def groq_invoke(prompt: str) -> str:
    try:
        response = client_groq.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise data analyst.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        content = response.choices[0].message.content
        if not content:
            return "fallback"

        return content.strip()
    except Exception as e:
        print("Groq API Error:", e)
        return "fallback"


st.set_page_config(
    page_title="Moonshot Luggage Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


DATA_DIR = Path("data/processed")
CLEANED_PATH = DATA_DIR / "cleaned.csv"
COMPETITIVE_PATH = DATA_DIR / "competitive_analysis.csv"
PRICING_PATH = DATA_DIR / "pricing_insights_final.csv"
COMPETITIVE_NOTES_PATH = DATA_DIR / "competitive_insights.txt"
PRICING_NOTES_PATH = DATA_DIR / "pricing_llm_insights.txt"
BRAND_INSIGHTS_PATH = Path("brand_insights.json")


# --- LLM-driven keyword/stopword/theme extraction with fallback ---
def get_llm_terms(prompt: str, fallback: set | dict) -> set | dict:
    """
    Query Groq LLM for a list/dict of terms. Fallback to hardcoded if LLM fails.
    """
    try:
        response = groq_invoke(prompt)
        # Try to parse as JSON (for dicts) or comma-separated (for sets)
        try:
            parsed = json.loads(response)
            if isinstance(fallback, dict) and isinstance(parsed, dict):
                # Convert all values to sets
                return {k: set(v) for k, v in parsed.items()}
            if isinstance(fallback, set) and isinstance(parsed, list):
                return set(parsed)
        except Exception:
            # Fallback: parse comma-separated for sets
            if isinstance(fallback, set):
                return set(map(str.strip, response.split(",")))
        return fallback
    except Exception:
        return fallback

# Fallback values (original hardcoded)
_POSITIVE_TERMS = {
    "sturdy", "durable", "smooth", "premium", "spacious", "lightweight", "excellent", "good", "great", "amazing", "value", "quality", "strong", "best", "comfortable", "recommended", "stylish", "easy", "perfect",
}
_NEGATIVE_TERMS = {
    "scratch", "scratches", "broken", "damage", "damaged", "poor", "bad", "heavy", "defect", "defective", "issue", "issues", "zip", "zipper", "wheel", "wheels", "lock", "small", "late", "weak", "problem",
}
_STOPWORDS = {
    "a", "about", "all", "an", "and", "are", "at", "be", "been", "but", "by", "for", "from", "good", "great", "had", "has", "have", "i", "if", "in", "is", "it", "its", "my", "not", "of", "on", "or", "our", "so", "that", "the", "their", "there", "this", "to", "very", "was", "were", "with", "you", "your",
}
_THEME_KEYWORDS = {
    "durability": {"durable", "sturdy", "strong", "solid", "hard", "quality"},
    "wheels": {"wheel", "wheels", "rolling", "smooth", "spinner"},
    "space": {"space", "spacious", "storage", "capacity", "room"},
    "design": {"design", "look", "stylish", "premium", "finish"},
    "weight": {"lightweight", "light", "heavy", "weight"},
    "lock": {"lock", "zip", "zipper", "security"},
    "value": {"value", "price", "worth", "budget"},
    "delivery": {"delivery", "packaging", "received", "late"},
}

# Prompts for LLM
POSITIVE_TERMS = get_llm_terms(
    "List the top 20 positive sentiment keywords (single words only, comma separated, no explanations) commonly found in Indian e-commerce luggage product reviews.",
    _POSITIVE_TERMS,
)
NEGATIVE_TERMS = get_llm_terms(
    "List the top 20 negative sentiment keywords (single words only, comma separated, no explanations) commonly found in Indian e-commerce luggage product reviews.",
    _NEGATIVE_TERMS,
)
STOPWORDS = get_llm_terms(
    "List the 30 most common English stopwords (single words only, comma separated, no explanations).",
    _STOPWORDS,
)
THEME_KEYWORDS = get_llm_terms(
    "Return a JSON dictionary mapping luggage review themes to a list of keywords. Themes: durability, wheels, space, design, weight, lock, value, delivery. Example: {\"durability\": [\"durable\", ...], ...}. Only output JSON.",
    _THEME_KEYWORDS,
)


def format_currency(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"Rs {value:,.0f}"


def format_percent(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.1f}%"


def format_number(value: float) -> str:
    if pd.isna(value):
        return "-"
    if value >= 1000:
        return f"{value:,.0f}"
    return f"{value:.2f}"


def clean_text(value: str) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return [token for token in tokens if token not in STOPWORDS]


def classify_sentiment(score: float) -> str:
    if pd.isna(score):
        return "Unknown"
    if score >= 4.2:
        return "Positive"
    if score >= 3.4:
        return "Neutral"
    return "Negative"


def extract_size(title: str) -> str:
    title = clean_text(title).lower()
    if any(token in title for token in ["cabin", "carry on", "carry-on", "55cm", "small"]):
        return "Cabin / Small"
    if any(token in title for token in ["medium", "24 inch", "24in", "m "]):
        return "Medium"
    if any(token in title for token in ["large", "28 inch", "28in", "check-in", "check in"]):
        return "Large"
    if any(token in title for token in ["set", "combo"]):
        return "Set / Combo"
    return "Unspecified"


def extract_category(title: str) -> str:
    title = clean_text(title).lower()
    if "duffel" in title or "duffle" in title:
        return "Duffel"
    if "backpack" in title:
        return "Backpack"
    if "hard" in title or "polypropylene" in title or "polycarbonate" in title:
        return "Hard Case"
    if "soft" in title or "fabric" in title:
        return "Soft Case"
    if "trolley" in title or "suitcase" in title:
        return "Trolley"
    return "Other"


def derive_discount(row: pd.Series) -> float:
    title = clean_text(row.get("product_title", ""))
    price = row.get("price")
    if pd.isna(price) or price <= 0:
        return math.nan

    match = re.search(r"(\d{1,2})\s*%\s*off", title.lower())
    if match:
        return float(match.group(1))

    if price < 2000:
        return 12.0
    if price < 3000:
        return 18.0
    if price < 4000:
        return 24.0
    return 30.0


def build_theme_counter(texts: pd.Series, theme_names: set[str] | None = None) -> Counter:
    joined = " ".join(clean_text(text) for text in texts if clean_text(text))
    counts = Counter()
    for theme, keywords in THEME_KEYWORDS.items():
        if theme_names and theme not in theme_names:
            continue
        hits = sum(joined.lower().count(keyword) for keyword in keywords)
        if hits:
            counts[theme] = hits
    return counts


def extract_top_terms(texts: pd.Series, lexicon: set[str], top_n: int = 5) -> list[str]:
    tokens = []
    for text in texts:
        words = tokenize(clean_text(text))
        tokens.extend(word for word in words if word in lexicon)
    return [word for word, _ in Counter(tokens).most_common(top_n)]


def synthesize_reviews(group: pd.DataFrame) -> str:
    avg_rating = group["review_rating"].mean()
    sentiment = classify_sentiment(avg_rating)
    top_praise = extract_top_terms(group["review_body"], POSITIVE_TERMS, top_n=3)
    top_complaints = extract_top_terms(group["review_body"], NEGATIVE_TERMS, top_n=3)

    praise_text = ", ".join(top_praise) if top_praise else "quality and usability"
    complaint_text = ", ".join(top_complaints) if top_complaints else "few recurring complaints"

    return (
        f"{sentiment} customer response with an average review rating of {avg_rating:.2f}. "
        f"Most appreciation centers on {praise_text}. "
        f"Main friction points are {complaint_text}."
    )


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str, dict]:
    reviews = pd.read_csv(CLEANED_PATH)
    reviews["price"] = pd.to_numeric(reviews["price"], errors="coerce")
    reviews["product_rating"] = pd.to_numeric(reviews["product_rating"], errors="coerce")
    reviews["review_rating"] = pd.to_numeric(reviews["review_rating"], errors="coerce")
    reviews["review_count"] = reviews.groupby("asin")["asin"].transform("count")
    reviews["sentiment_score"] = (reviews["review_rating"] / 5.0) * 100
    reviews["sentiment_label"] = reviews["review_rating"].apply(classify_sentiment)
    reviews["size_bucket"] = reviews["product_title"].apply(extract_size)
    reviews["category_bucket"] = reviews["product_title"].apply(extract_category)
    reviews["discount_pct"] = reviews.apply(derive_discount, axis=1)
    reviews["review_body"] = reviews["review_body"].fillna("")
    reviews["review_title"] = reviews["review_title"].fillna("")

    competitive = pd.read_csv(COMPETITIVE_PATH) if COMPETITIVE_PATH.exists() else pd.DataFrame()
    pricing = pd.read_csv(PRICING_PATH) if PRICING_PATH.exists() else pd.DataFrame()
    competitive_notes = COMPETITIVE_NOTES_PATH.read_text(encoding="utf-8") if COMPETITIVE_NOTES_PATH.exists() else ""
    pricing_notes = PRICING_NOTES_PATH.read_text(encoding="utf-8") if PRICING_NOTES_PATH.exists() else ""
    brand_insights = {}
    if BRAND_INSIGHTS_PATH.exists():
        with open(BRAND_INSIGHTS_PATH, "r", encoding="utf-8") as file:
            brand_insights = json.load(file)
    return reviews, competitive, pricing, competitive_notes, pricing_notes, brand_insights


def build_brand_summary(filtered_reviews: pd.DataFrame, brand_insights: dict) -> pd.DataFrame:
    brand_summary = (
        filtered_reviews.groupby("brand")
        .agg(
            avg_price=("price", "mean"),
            avg_discount_pct=("discount_pct", "mean"),
            avg_star_rating=("product_rating", "mean"),
            review_count=("asin", "count"),
            product_count=("asin", "nunique"),
            sentiment_score=("sentiment_score", "mean"),
        )
        .reset_index()
    )

    pros = []
    cons = []
    top_themes = []
    llm_summaries = []
    llm_sentiments = []
    llm_labels = []
    for brand in brand_summary["brand"]:
        subset = filtered_reviews[filtered_reviews["brand"] == brand]
        insight = brand_insights.get(brand, {})

        llm_praise = insight.get("common_praise", [])
        llm_complaints = insight.get("common_complaints", [])
        llm_positive_themes = insight.get("positive_themes", [])
        llm_negative_themes = insight.get("negative_themes", [])
        llm_summary = insight.get("summary", "")
        llm_score = insight.get("sentiment_score")
        llm_label = insight.get("overall_sentiment", "")

        fallback_pros = ", ".join(extract_top_terms(subset["review_body"], POSITIVE_TERMS, top_n=3)) or "No dominant pros"
        fallback_cons = ", ".join(extract_top_terms(subset["review_body"], NEGATIVE_TERMS, top_n=3)) or "No dominant cons"
        fallback_themes = ", ".join(theme for theme, _ in build_theme_counter(subset["review_body"]).most_common(3)) or "Mixed feedback"

        pros.append(", ".join(llm_praise) if llm_praise else fallback_pros)
        cons.append(", ".join(llm_complaints) if llm_complaints else fallback_cons)

        merged_themes = llm_positive_themes + llm_negative_themes
        top_themes.append(", ".join(merged_themes[:4]) if merged_themes else fallback_themes)
        llm_summaries.append(llm_summary or synthesize_reviews(subset))
        llm_labels.append(llm_label or classify_sentiment(subset["review_rating"].mean()))

        if isinstance(llm_score, (int, float)):
            llm_sentiments.append((float(llm_score) / 5.0) * 100)
        else:
            llm_sentiments.append(subset["sentiment_score"].mean())

    brand_summary["top_pros"] = pros
    brand_summary["top_cons"] = cons
    brand_summary["top_themes"] = top_themes
    brand_summary["brand_summary"] = llm_summaries
    brand_summary["llm_sentiment_label"] = llm_labels
    brand_summary["sentiment_score"] = llm_sentiments
    return brand_summary.sort_values(["sentiment_score", "avg_star_rating"], ascending=False)


def build_product_summary(filtered_reviews: pd.DataFrame) -> pd.DataFrame:
    product_base = (
        filtered_reviews.groupby(["asin", "brand", "product_title"])
        .agg(
            price=("price", "mean"),
            product_rating=("product_rating", "mean"),
            avg_review_rating=("review_rating", "mean"),
            review_count=("asin", "count"),
            category_bucket=("category_bucket", "first"),
            size_bucket=("size_bucket", "first"),
            discount_pct=("discount_pct", "mean"),
            url=("url", "first"),
        )
        .reset_index()
    )

    product_base["list_price"] = product_base["price"] / (1 - (product_base["discount_pct"] / 100)).replace(0, pd.NA)
    product_base["sentiment_label"] = product_base["avg_review_rating"].apply(classify_sentiment)
    product_base["sentiment_score"] = (product_base["avg_review_rating"] / 5.0) * 100

    syntheses = []
    complaint_themes = []
    appreciation_themes = []
    for asin in product_base["asin"]:
        group = filtered_reviews[filtered_reviews["asin"] == asin]
        syntheses.append(synthesize_reviews(group))
        negative_themes = build_theme_counter(group.loc[group["review_rating"] < 4, "review_body"])
        positive_themes = build_theme_counter(group.loc[group["review_rating"] >= 4, "review_body"])
        complaint_themes.append(", ".join(theme for theme, _ in negative_themes.most_common(3)) or "No strong complaints")
        appreciation_themes.append(", ".join(theme for theme, _ in positive_themes.most_common(3)) or "No dominant appreciation theme")

    product_base["review_synthesis"] = syntheses
    product_base["top_complaint_themes"] = complaint_themes
    product_base["top_appreciation_themes"] = appreciation_themes
    return product_base.sort_values(["sentiment_score", "review_count"], ascending=False)


def apply_filters(reviews: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("## Control Room")
    st.sidebar.caption("Click filters to reshape every chart, KPI, and table in real time.")

    brands = sorted(reviews["brand"].dropna().unique().tolist())
    categories = sorted(reviews["category_bucket"].dropna().unique().tolist())
    sizes = sorted(reviews["size_bucket"].dropna().unique().tolist())
    sentiments = sorted(reviews["sentiment_label"].dropna().unique().tolist())

    selected_brands = st.sidebar.multiselect("Brand selector", options=brands, default=brands)
    price_min = float(reviews["price"].min())
    price_max = float(reviews["price"].max())
    selected_price = st.sidebar.slider("Price range", min_value=price_min, max_value=price_max, value=(price_min, price_max), step=100.0)
    min_rating = st.sidebar.slider("Minimum rating", min_value=1.0, max_value=5.0, value=3.5, step=0.1)
    selected_categories = st.sidebar.multiselect("Luggage category", options=categories, default=categories)
    selected_sizes = st.sidebar.multiselect("Size filter", options=sizes, default=sizes)
    selected_sentiments = st.sidebar.multiselect("Sentiment filter", options=sentiments, default=sentiments)

    filtered = reviews[
        reviews["brand"].isin(selected_brands)
        & reviews["price"].between(selected_price[0], selected_price[1], inclusive="both")
        & (reviews["product_rating"] >= min_rating)
        & reviews["category_bucket"].isin(selected_categories)
        & reviews["size_bucket"].isin(selected_sizes)
        & reviews["sentiment_label"].isin(selected_sentiments)
    ].copy()

    return filtered


def render_kpis(filtered_reviews: pd.DataFrame) -> None:
    brand_count = filtered_reviews["brand"].nunique()
    product_count = filtered_reviews["asin"].nunique()
    review_count = len(filtered_reviews)
    avg_sentiment = filtered_reviews["sentiment_score"].mean()

    cols = st.columns(4)
    cols[0].metric("Total brands tracked", format_number(brand_count))
    cols[1].metric("Total products analyzed", format_number(product_count))
    cols[2].metric("Total reviews analyzed", format_number(review_count))
    cols[3].metric("Average sentiment snapshot", f"{avg_sentiment:.1f}/100")


def render_overview(filtered_reviews: pd.DataFrame, brand_summary: pd.DataFrame) -> None:
    st.markdown("## Dashboard Overview")
    render_kpis(filtered_reviews)

    chart_col, dist_col = st.columns([1.2, 1])
    with chart_col:
        fig = px.bar(
            brand_summary,
            x="brand",
            y="review_count",
            color="sentiment_score",
            color_continuous_scale=["#b42318", "#f5c451", "#157f3b"],
            text_auto=".0f",
            title="Review Volume by Brand",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with dist_col:
        sentiment_mix = (
            filtered_reviews["sentiment_label"].value_counts().rename_axis("sentiment_label").reset_index(name="count")
        )
        fig = px.pie(
            sentiment_mix,
            names="sentiment_label",
            values="count",
            hole=0.58,
            color="sentiment_label",
            color_discrete_map={
                "Positive": "#157f3b",
                "Neutral": "#f5c451",
                "Negative": "#b42318",
                "Unknown": "#687076",
            },
            title="Sentiment Mix",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    price_col, scatter_col = st.columns(2)
    with price_col:
        price_data = (
            filtered_reviews.groupby("brand")
            .agg(avg_price=("price", "mean"), avg_discount_pct=("discount_pct", "mean"))
            .reset_index()
        )
        fig = px.bar(
            price_data,
            x="brand",
            y="avg_price",
            color="avg_discount_pct",
            color_continuous_scale=["#dfe9e2", "#7aa874", "#173f2d"],
            title="Pricing Overview",
            labels={"avg_price": "Average price", "brand": "Brand", "avg_discount_pct": "Average discount %"},
        )
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with scatter_col:
        fig = px.scatter(
            brand_summary,
            x="avg_price",
            y="avg_star_rating",
            size="review_count",
            color="brand",
            hover_data=["avg_discount_pct", "sentiment_score", "product_count"],
            title="Brand Positioning: Price vs Rating",
            labels={"avg_price": "Average price", "avg_star_rating": "Average star rating"},
        )
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)


def render_brand_comparison(brand_summary: pd.DataFrame, competitive_notes: str, pricing_notes: str) -> None:
    st.markdown("## Brand Comparison View")
    left, right = st.columns([1.5, 1])

    display_table = brand_summary.copy()
    display_table["avg_price"] = display_table["avg_price"].map(format_currency)
    display_table["avg_discount_pct"] = display_table["avg_discount_pct"].map(format_percent)
    display_table["avg_star_rating"] = display_table["avg_star_rating"].map(lambda x: f"{x:.2f}")
    display_table["sentiment_score"] = display_table["sentiment_score"].map(lambda x: f"{x:.1f}/100")
    display_table = display_table.rename(
        columns={
            "brand": "Brand",
            "avg_price": "Average price",
            "avg_discount_pct": "Average discount %",
            "avg_star_rating": "Average star rating",
            "review_count": "Review count",
            "product_count": "Product count",
            "sentiment_score": "Sentiment score",
            "llm_sentiment_label": "Overall sentiment",
            "top_pros": "Top pros",
            "top_cons": "Top cons",
            "top_themes": "Top themes",
            "brand_summary": "Brand summary",
        }
    )

    with left:
        st.dataframe(display_table, use_container_width=True, hide_index=True)

    with right:
        leader = brand_summary.iloc[0] if not brand_summary.empty else None
        laggard = brand_summary.iloc[-1] if not brand_summary.empty else None
        st.markdown("### Snapshot")
        if leader is not None:
            st.markdown(
                f"""
                <div class="insight-card">
                    <p class="eyebrow">Best current performer</p>
                    <h3>{leader['brand']}</h3>
                    <p>{leader['brand_summary']}</p>
                </div>
                <div class="insight-card muted">
                    <p class="eyebrow">Watchout brand</p>
                    <h3>{laggard['brand']}</h3>
                    <p>{laggard['brand_summary']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if competitive_notes:
            st.markdown("### Competitive insight")
            st.write(competitive_notes)
        if pricing_notes:
            st.markdown("### Pricing insight")
            st.write(pricing_notes)


def render_product_drilldown(product_summary: pd.DataFrame, filtered_reviews: pd.DataFrame) -> None:
    st.markdown("## Product Drilldown")
    if product_summary.empty:
        st.warning("No products match the active filters.")
        return

    product_options = product_summary.apply(lambda row: f"{row['brand']} | {row['product_title'][:95]}", axis=1).tolist()
    selected_label = st.selectbox("Choose a product", options=product_options)
    selected_index = product_options.index(selected_label)
    product = product_summary.iloc[selected_index]

    product_reviews = filtered_reviews[filtered_reviews["asin"] == product["asin"]].copy()

    metric_cols = st.columns(6)
    metric_cols[0].metric("Price", format_currency(product["price"]))
    metric_cols[1].metric("List price", format_currency(product["list_price"]))
    metric_cols[2].metric("Discount", format_percent(product["discount_pct"]))
    metric_cols[3].metric("Rating", f"{product['product_rating']:.2f}")
    metric_cols[4].metric("Review count", format_number(product["review_count"]))
    metric_cols[5].metric("Sentiment", f"{product['sentiment_score']:.1f}/100")

    st.markdown(
        f"""
        <div class="product-panel">
            <p class="eyebrow">{product['brand']} • {product['category_bucket']} • {product['size_bucket']}</p>
            <h3>{product['product_title']}</h3>
            <p>{product['review_synthesis']}</p>
            <p><a href="{product['url']}" target="_blank">Open product page</a></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    theme_col, review_col = st.columns([1, 1.3])
    with theme_col:
        st.markdown("### Themes")
        st.write(f"Top complaint themes: {product['top_complaint_themes']}")
        st.write(f"Top appreciation themes: {product['top_appreciation_themes']}")

        rating_breakdown = (
            product_reviews["review_rating"].value_counts().sort_index().rename_axis("rating").reset_index(name="count")
        )
        fig = px.bar(
            rating_breakdown,
            x="rating",
            y="count",
            title="Review Rating Breakdown",
            labels={"rating": "Review rating", "count": "Count"},
            color="count",
            color_continuous_scale=["#e7f1ea", "#2e6b47"],
        )
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with review_col:
        st.markdown("### Recent review snippets")
        review_table = product_reviews[["review_rating", "review_title", "review_body"]].copy()
        review_table["review_body"] = review_table["review_body"].apply(lambda text: clean_text(text)[:220] + ("..." if len(clean_text(text)) > 220 else ""))
        review_table["review_title"] = review_table["review_title"].replace("", "Untitled review")
        st.dataframe(review_table.head(12), use_container_width=True, hide_index=True)


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ms-ink: #14261d;
            --ms-ink-soft: #335245;
            --ms-paper: rgba(255, 251, 244, 0.9);
            --ms-paper-strong: rgba(255, 252, 246, 0.96);
            --ms-border: rgba(23, 63, 45, 0.12);
            --ms-accent: #173f2d;
            --ms-link: #0d5f3b;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(29, 78, 60, 0.22), transparent 30%),
                radial-gradient(circle at top right, rgba(197, 162, 82, 0.18), transparent 28%),
                linear-gradient(180deg, #f6f2e9 0%, #efe7d7 100%);
            color: var(--ms-ink);
        }
        .stApp,
        .stApp p,
        .stApp label,
        .stApp span,
        .stApp div,
        .stApp li {
            color: var(--ms-ink);
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #163728 0%, #214f3a 100%);
        }
        [data-testid="stSidebar"] * {
            color: #f4efe6;
        }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] div {
            color: #f4efe6 !important;
        }
        .hero {
            background: linear-gradient(135deg, rgba(23, 63, 45, 0.96), rgba(78, 110, 72, 0.94));
            color: #f6f2e9;
            padding: 1.5rem 1.6rem;
            border-radius: 22px;
            border: 1px solid rgba(246, 242, 233, 0.18);
            box-shadow: 0 18px 40px rgba(20, 38, 29, 0.16);
            margin-bottom: 1.2rem;
        }
        .hero h1, .hero p, .hero span {
            color: #f6f2e9;
        }
        .eyebrow {
            text-transform: uppercase;
            letter-spacing: 0.12rem;
            font-size: 0.76rem;
            opacity: 0.72;
            margin-bottom: 0.35rem;
        }
        .insight-card, .product-panel {
            background: var(--ms-paper-strong);
            border: 1px solid var(--ms-border);
            border-radius: 18px;
            padding: 1rem 1rem 0.8rem 1rem;
            margin-bottom: 0.9rem;
            box-shadow: 0 10px 24px rgba(20, 38, 29, 0.07);
        }
        .insight-card *,
        .product-panel *,
        .insight-card h1,
        .insight-card h2,
        .insight-card h3,
        .product-panel h1,
        .product-panel h2,
        .product-panel h3,
        .insight-card p,
        .product-panel p,
        .insight-card a,
        .product-panel a {
            color: var(--ms-ink) !important;
        }
        .insight-card.muted {
            background: rgba(244, 236, 222, 0.8);
        }
        [data-testid="stMetric"] {
            background: var(--ms-paper);
            border: 1px solid var(--ms-border);
            padding: 0.85rem;
            border-radius: 18px;
            box-shadow: 0 10px 24px rgba(20, 38, 29, 0.05);
        }
        [data-testid="stMetric"] *,
        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] *,
        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] * {
            color: var(--ms-ink) !important;
        }
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stText"],
        .stSelectbox label,
        .stMultiSelect label,
        .stSlider label,
        .stNumberInput label {
            color: var(--ms-ink) !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 251, 244, 0.55);
            border: 1px solid var(--ms-border);
            border-radius: 999px;
            color: var(--ms-ink) !important;
            padding: 0.45rem 0.9rem;
        }
        .stTabs [aria-selected="true"] {
            background: #173f2d !important;
            border-color: #173f2d !important;
            color: #f6f2e9 !important;
        }
        .stTabs [aria-selected="true"] * {
            color: #f6f2e9 !important;
        }
        h1, h2, h3 {
            color: var(--ms-accent);
        }
        a {
            color: var(--ms-link) !important;
        }
        [data-testid="stDataFrame"] *,
        [data-testid="stTable"] * {
            color: var(--ms-ink) !important;
        }
        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            background: rgba(255, 252, 246, 0.7);
            border-radius: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    apply_theme()
    reviews, _competitive, _pricing, competitive_notes, pricing_notes, brand_insights = load_data()
    filtered_reviews = apply_filters(reviews)

    st.markdown(
        """
        <div class="hero">
            <p class="eyebrow">Moonshot Dashboard</p>
            <h1>Luggage Market Intelligence</h1>
            <p>Track brand performance, compare price positioning, and drill into product-level customer sentiment with click-driven analysis.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if filtered_reviews.empty:
        st.warning("No rows match the current filters. Relax one or two controls to bring products back into view.")
        return

    brand_summary = build_brand_summary(filtered_reviews, brand_insights)
    product_summary = build_product_summary(filtered_reviews)

    overview_tab, brand_tab, product_tab = st.tabs(
        ["Dashboard Overview", "Brand Comparison", "Product Drilldown"]
    )

    with overview_tab:
        render_overview(filtered_reviews, brand_summary)

    with brand_tab:
        render_brand_comparison(brand_summary, competitive_notes, pricing_notes)

    with product_tab:
        render_product_drilldown(product_summary, filtered_reviews)


if __name__ == "__main__":
    main()
