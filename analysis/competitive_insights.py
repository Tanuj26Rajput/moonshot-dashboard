import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from openai import OpenAI
import os
from dotenv import load_dotenv

# ---------------- GROQ SETUP ---------------- #
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


# ---------------- LOAD DATA ---------------- #
df = pd.read_csv("data/processed/cleaned.csv")

# ---------------- CLEAN DATA ---------------- #
df["price"] = df["price"].astype(str).str.replace(",", "").astype(float)

# If discount not present → assume (you can remove if real exists)
if "discount" not in df.columns:
    df["discount"] = np.random.uniform(15, 35, size=len(df))

# ---------------- SENTIMENT SCORE ---------------- #
sentiment_map = {
    "positive": 1,
    "neutral": 0,
    "negative": -1
}

if "sentiment" in df.columns:
    df["sentiment_score"] = df["sentiment"].map(sentiment_map)
else:
    # fallback using rating
    df["sentiment_score"] = df["review_rating"] / 5


# ---------------- AGGREGATION ---------------- #

summary = df.groupby("brand").agg({
    "price": "mean",
    "discount": "mean",
    "review_rating": "mean",
    "asin": "count",
    "sentiment_score": "mean"
}).reset_index()

summary.rename(columns={
    "price": "avg_price",
    "discount": "avg_discount",
    "review_rating": "avg_rating",
    "asin": "review_count",
    "sentiment_score": "avg_sentiment"
}, inplace=True)


# ---------------- NORMALIZATION ---------------- #

scaler = MinMaxScaler()

summary[[
    "price_norm",
    "discount_norm",
    "rating_norm",
    "review_norm",
    "sentiment_norm"
]] = scaler.fit_transform(summary[[
    "avg_price",
    "avg_discount",
    "avg_rating",
    "review_count",
    "avg_sentiment"
]])


# ---------------- FINAL COMPETITION SCORE ---------------- #

summary["competition_score"] = (
    (summary["rating_norm"] * 0.30) +
    (summary["review_norm"] * 0.20) +
    (summary["sentiment_norm"] * 0.20) +
    (summary["discount_norm"] * 0.15) -
    (summary["price_norm"] * 0.15)
)

summary = summary.sort_values(by="competition_score", ascending=False)


# ---------------- PRINT TABLE ---------------- #

print("\n🏆 COMPETITIVE ANALYSIS:\n")
print(summary[[
    "brand",
    "avg_price",
    "avg_discount",
    "avg_rating",
    "review_count",
    "avg_sentiment",
    "competition_score"
]])


# ---------------- LLM INSIGHTS ---------------- #

def build_competition_prompt(summary_df):
    return f"""
        You are a senior business analyst.

        Here is competitive data of luggage brands:

        {summary_df.to_dict(orient='records')}

        Tasks:
        1. Identify which brand is WINNING overall and why.
        2. Identify weakest brand and why.
        3. Compare brands on:
           - price vs quality
           - discount strategy
           - customer satisfaction
        4. Identify:
           - Best budget brand
           - Best premium brand
           - Best overall value brand
        5. Highlight any surprising insights.

        Keep answer sharp, structured, and decision-focused.
    """

llm_output = groq_invoke(build_competition_prompt(summary))

print("\n🧠 LLM COMPETITIVE INSIGHTS:\n")
print(llm_output)


# ---------------- SAVE OUTPUT ---------------- #

summary.to_csv("data/processed/competitive_analysis.csv", index=False)

with open("data/processed/competitive_insights.txt", "w", encoding="utf-8") as f:
    f.write(llm_output)