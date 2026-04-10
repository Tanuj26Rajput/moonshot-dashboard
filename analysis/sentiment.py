import json
import os

import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
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
                    "content": "You are a precise data analyst. Always return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if not content:
            return "fallback"

        return content.strip()
    except Exception as e:
        print("Groq API Error:", e)
        return "fallback"


df = pd.read_csv("data/processed/cleaned.csv")
brand_groups = df.groupby("brand")


def build_prompt(brand, reviews, ratings):
    combined = [
        f"{review} (Rating: {rating})"
        for review, rating in zip(reviews[:60], ratings[:60])
    ]
    sample_reviews = "\n".join(combined)

    prompt = PromptTemplate.from_template(
        """
        You are a data analyst specializing in sentiment analysis for product reviews.
        Analyze the following reviews for the brand {brand} and provide insights on the overall sentiment, key themes, and common feedback.

        Reviews:
        {sample_reviews}

        Return a valid JSON object with exactly these keys:
        {{
          "overall_sentiment": "Positive | Negative | Neutral | Mixed",
          "sentiment_score": 1,
          "positive_themes": ["theme 1"],
          "negative_themes": ["theme 1"],
          "common_complaints": ["complaint 1"],
          "common_praise": ["praise 1"],
          "summary": "short summary"
        }}

        Rules:
        - sentiment_score must be an integer from 1 to 5.
        - Keep list items concise.
        - Do not include markdown fences or any text outside the JSON object.
        """
    )
    return prompt.format(brand=brand, sample_reviews=sample_reviews)


def parse_response(response: str) -> dict:
    try:
        return json.loads(response)
    except Exception:
        print("Failed to parse response as JSON. Returning raw text.")
        return {"raw_response": response}


brand_insights = {}
for brand, group in brand_groups:
    reviews = group["review_body"].dropna().tolist()
    ratings = group["review_rating"].dropna().tolist()

    if not reviews:
        continue

    prompt = build_prompt(brand, reviews, ratings)
    response = groq_invoke(prompt)
    response_data = parse_response(response)

    if response_data:
        brand_insights[brand] = response_data
    else:
        brand_insights[brand] = {"raw_response": response}

with open("brand_insights.json", "w", encoding="utf-8") as f:
    json.dump(brand_insights, f, indent=4, ensure_ascii=False)
