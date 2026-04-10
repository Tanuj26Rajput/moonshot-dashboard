import pandas as pd
import re

import re

def clean_price(x):
    if pd.isna(x):
        return None
    
    x = str(x)
    
    # Remove everything except digits
    x = re.sub(r"[^\d]", "", x)
    
    return float(x) if x != "" else None

df = pd.read_csv("data/raw/amazon_luggage.csv")

# Clean price
df["price"] = df["price"].apply(clean_price)

# Clean product rating
df["product_rating"] = df["product_rating"].str.extract(r"(\d+\.\d+)").astype(float)

# Clean review rating
df["review_rating"] = df["review_rating"].str.extract(r"(\d+\.\d+)").astype(float)

# Drop null reviews
df = df.dropna(subset=["review_body"])

# Remove duplicates (IMPORTANT)
df = df.drop_duplicates(subset=["review_body"])

# Add review length (useful later)
df["review_length"] = df["review_body"].apply(len)

df.to_csv("data/processed/cleaned.csv", index=False)