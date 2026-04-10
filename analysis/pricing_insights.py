import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


# ---------------- CORE METRICS ---------------- #

summary = df.groupby("brand").agg({
    "price": "mean",
    "review_rating": "mean",
    "asin": "count"
}).reset_index()

summary.rename(columns={
    "price": "avg_price",
    "review_rating": "avg_rating",
    "asin": "product_count"
}, inplace=True)

# Price spread
spread = df.groupby("brand")["price"].agg(["min", "max"]).reset_index()
summary = summary.merge(spread, on="brand")


# ---------------- POSITIONING ---------------- #

def classify(price):
    if price < 2000:
        return "Budget"
    elif price < 3000:
        return "Mid-range"
    else:
        return "Premium"

summary["positioning"] = summary["avg_price"].apply(classify)


# ---------------- NORMALIZATION ---------------- #

scaler = MinMaxScaler()
summary[["price_norm", "rating_norm", "count_norm"]] = scaler.fit_transform(
    summary[["avg_price", "avg_rating", "product_count"]]
)


# ---------------- FINAL SCORE ---------------- #

summary["value_score"] = (
    (summary["rating_norm"] * 0.5) +
    (summary["count_norm"] * 0.3) -
    (summary["price_norm"] * 0.2)
)

summary = summary.sort_values(by="value_score", ascending=False)


# ---------------- LLM INSIGHTS ---------------- #

def build_pricing_prompt(summary_df):
    return f"""
You are a senior market analyst.

Here is structured pricing data for luggage brands:

{summary_df[['brand','avg_price','avg_rating','product_count','min','max','positioning','value_score']].to_dict(orient='records')}

Tasks:
1. Identify budget, mid-range, and premium brands.
2. Explain pricing differences across brands.
3. Identify which brand offers BEST VALUE FOR MONEY and why.
4. Identify which brand is overpriced (high price but not strong rating/value).
5. Highlight any interesting patterns in price spread.
6. Give final recommendation:
   - Best budget brand
   - Best premium brand
   - Overall best brand

Keep answer crisp, structured, and insightful.
"""

llm_insights = groq_invoke(build_pricing_prompt(summary))


# ---------------- PRINT OUTPUT ---------------- #

print("\n🔥 FINAL PRICING SUMMARY:\n")
print(summary)

print("\n🧠 LLM INSIGHTS:\n")
print(llm_insights)


# ---------------- VISUALIZATION ---------------- #

# 1️⃣ Avg Price
plt.figure()
plt.bar(summary["brand"], summary["avg_price"])
plt.xticks(rotation=45)
plt.ylabel("Average Price (₹)")
plt.title("Average Selling Price by Brand")
plt.show()


# 2️⃣ Price Spread
plt.figure()
df.boxplot(column="price", by="brand")
plt.suptitle("")
plt.title("Product Price Spread")
plt.xticks(rotation=45)
plt.show()


# 3️⃣ Price vs Rating
plt.figure()
plt.scatter(summary["avg_price"], summary["avg_rating"])

for i, row in summary.iterrows():
    plt.text(row["avg_price"], row["avg_rating"], row["brand"])

plt.xlabel("Average Price (₹)")
plt.ylabel("Average Rating")
plt.title("Price vs Rating Positioning")
plt.show()


# 4️⃣ Value Score Ranking
plt.figure()
plt.bar(summary["brand"], summary["value_score"])
plt.xticks(rotation=45)
plt.ylabel("Score")
plt.title("Best Value for Money Ranking")
plt.show()


# ---------------- SAVE OUTPUT ---------------- #

summary.to_csv("data/processed/pricing_insights_final.csv", index=False)

with open("data/processed/pricing_llm_insights.txt", "w", encoding="utf-8") as f:
    f.write(llm_insights)