# Moonshot Dashboard

A Streamlit-based luggage market intelligence dashboard that combines scraped Amazon product data, cleaned review datasets, Groq-powered sentiment analysis, pricing insights, and competitive benchmarking.

The project is designed to answer questions like:
- Which luggage brand is performing best overall?
- Which brands offer the strongest value for money?
- What do customers praise most often?
- What complaints repeat across brands and products?
- How do pricing, ratings, and review sentiment shift across the market?

## What This Project Does

This repository turns raw e-commerce review data into an interactive dashboard with:
- brand-level comparison
- pricing overview
- product drilldowns
- review synthesis
- sentiment-driven summaries
- Groq-generated insight panels

The workflow is:
1. scrape luggage listings and reviews
2. clean and normalize the raw data
3. generate pricing and competition summaries
4. generate brand sentiment insights with Groq
5. render everything in a Streamlit dashboard

## Features

### Dashboard overview
- Total brands tracked
- Total products analyzed
- Total reviews analyzed
- Average sentiment snapshot
- Review volume by brand
- Sentiment mix
- Pricing overview
- Brand positioning by price vs rating

### Brand comparison
- Average price
- Average discount percentage
- Average star rating
- Review count
- Product count
- Sentiment score
- Top pros
- Top cons
- Top themes
- Brand summary from Groq

### Product drilldown
- Product title
- Price
- Estimated list price
- Discount percentage
- Rating
- Review count
- Review synthesis
- Top complaint themes
- Top appreciation themes
- Review snippets

### Filters and interactions
- Brand selector
- Price range filter
- Minimum rating filter
- Luggage category filter
- Size filter
- Sentiment filter

## Tech Stack

- `Streamlit` for the dashboard UI
- `Pandas` for data manipulation
- `Plotly` for interactive charts
- `OpenAI-compatible client` for Groq API calls
- `python-dotenv` for environment configuration
- `scikit-learn` for normalization and scoring logic
- `LangChain Core` for prompt templating in sentiment generation

## Project Structure

```text
moonshot-dashboard/
├── app.py
├── requirements.txt
├── brand_insights.json
├── analysis/
│   ├── sentiment.py
│   ├── pricing_insights.py
│   └── competitive_insights.py
├── data/
│   ├── raw/
│   │   └── amazon_luggage.csv
│   └── processed/
│       ├── cleaned.csv
│       ├── competitive_analysis.csv
│       ├── competitive_insights.txt
│       ├── pricing_insights_final.csv
│       └── pricing_llm_insights.txt
├── scraper/
│   └── amazon_scraper.py
└── util/
    └── data_cleaning.py
```

## Data Pipeline

### 1. Scraping
[amazon_scraper.py](/c:/Users/Tanuj%20Rajput/OneDrive/Documents/GitHub/moonshot-dashboard/scraper/amazon_scraper.py) collects luggage product and review data from Amazon and stores it in the raw dataset.

Expected raw fields include:
- brand
- asin
- product title
- price
- product rating
- review rating
- review title
- review body
- product URL

### 2. Data cleaning
[data_cleaning.py](/c:/Users/Tanuj%20Rajput/OneDrive/Documents/GitHub/moonshot-dashboard/util/data_cleaning.py) transforms the raw CSV into a clean analytical dataset by:
- cleaning price values
- extracting numeric product ratings
- extracting numeric review ratings
- dropping null review bodies
- removing duplicate reviews
- adding review length

Output:
- [cleaned.csv](/c:/Users/Tanuj%20Rajput/OneDrive/Documents/GitHub/moonshot-dashboard/data/processed/cleaned.csv)

### 3. Sentiment analysis
[sentiment.py](/c:/Users/Tanuj%20Rajput/OneDrive/Documents/GitHub/moonshot-dashboard/analysis/sentiment.py) uses Groq to generate brand-level sentiment summaries from review text.

Output:
- [brand_insights.json](/c:/Users/Tanuj%20Rajput/OneDrive/Documents/GitHub/moonshot-dashboard/brand_insights.json)

Each brand record includes:
- `overall_sentiment`
- `sentiment_score`
- `positive_themes`
- `negative_themes`
- `common_complaints`
- `common_praise`
- `summary`

### 4. Pricing analysis
[pricing_insights.py](/c:/Users/Tanuj%20Rajput/OneDrive/Documents/GitHub/moonshot-dashboard/analysis/pricing_insights.py) computes:
- average brand price
- average rating
- product count
- min and max price spread
- pricing segment classification
- value score

It also asks Groq for a pricing interpretation and recommendations.

Outputs:
- [pricing_insights_final.csv](/c:/Users/Tanuj%20Rajput/OneDrive/Documents/GitHub/moonshot-dashboard/data/processed/pricing_insights_final.csv)
- [pricing_llm_insights.txt](/c:/Users/Tanuj%20Rajput/OneDrive/Documents/GitHub/moonshot-dashboard/data/processed/pricing_llm_insights.txt)

### 5. Competitive analysis
[competitive_insights.py](/c:/Users/Tanuj%20Rajput/OneDrive/Documents/GitHub/moonshot-dashboard/analysis/competitive_insights.py) computes a competition score using:
- average price
- discount
- rating
- review count
- sentiment score

It then generates an LLM-based competitive summary.

Outputs:
- [competitive_analysis.csv](/c:/Users/Tanuj%20Rajput/OneDrive/Documents/GitHub/moonshot-dashboard/data/processed/competitive_analysis.csv)
- [competitive_insights.txt](/c:/Users/Tanuj%20Rajput/OneDrive/Documents/GitHub/moonshot-dashboard/data/processed/competitive_insights.txt)

## Dashboard Logic

[app.py](/c:/Users/Tanuj%20Rajput/OneDrive/Documents/GitHub/moonshot-dashboard/app.py) is the main application entry point.

The dashboard:
- loads cleaned product and review data
- reads precomputed pricing and competitive outputs
- consumes `brand_insights.json` from the sentiment pipeline
- uses Groq for product-level review synthesis and theme extraction when available
- falls back gracefully when Groq or Plotly are unavailable

### Groq usage in the dashboard

The app uses Groq in two ways:

1. Precomputed brand insights
   - generated by `analysis/sentiment.py`
   - stored in `brand_insights.json`
   - reused directly inside the dashboard

2. Live product insight generation
   - the dashboard can call Groq to synthesize product-level reviews
   - if Groq is not available, the app falls back to local heuristic summaries

This design gives you:
- richer AI summaries when credentials are available
- a still-functional dashboard in deployment environments with missing secrets or network restrictions

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd moonshot-dashboard
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

Notes:
- `GROQ_API_KEY` is required for LLM-generated sentiment and product synthesis
- `GROQ_MODEL` is optional and defaults to `llama-3.3-70b-versatile`

## Running The Project

### Step 1. Clean the raw data

```bash
python util/data_cleaning.py
```

### Step 2. Generate brand sentiment insights

```bash
python analysis/sentiment.py
```

### Step 3. Generate pricing insights

```bash
python analysis/pricing_insights.py
```

### Step 4. Generate competitive insights

```bash
python analysis/competitive_insights.py
```

### Step 5. Launch the dashboard

```bash
streamlit run app.py
```

## Deployment Notes

This project is compatible with Streamlit deployment platforms.

### Required deployment settings
- Add `GROQ_API_KEY` as an environment secret
- Ensure `requirements.txt` is present in the repo root

### Dependency fallback behavior

The app is designed to degrade gracefully:
- if `plotly` is missing, it falls back to Streamlit-native charts and tables
- if Groq is unavailable, it falls back to local heuristics and previously generated outputs

### Common deployment issue

If you see:

```text
ModuleNotFoundError: No module named 'plotly'
```

It usually means the deployment platform did not install dependencies from `requirements.txt`. Trigger a clean rebuild or redeploy after pushing the latest changes.

## Outputs Produced By The Project

| File | Purpose |
|---|---|
| `data/processed/cleaned.csv` | Cleaned review dataset |
| `brand_insights.json` | Groq-generated brand sentiment insights |
| `data/processed/pricing_insights_final.csv` | Structured pricing analysis |
| `data/processed/pricing_llm_insights.txt` | Groq-generated pricing narrative |
| `data/processed/competitive_analysis.csv` | Structured competition scoring |
| `data/processed/competitive_insights.txt` | Groq-generated competitive narrative |

## Limitations

- The current dataset is Amazon-specific
- Some discount and list-price values are estimated when explicit values are unavailable
- Product categories and sizes are partially inferred from title text
- Live Groq-powered product summaries may be slower than local heuristics
- Without API access, the dashboard relies on fallback summaries rather than fresh model outputs

## Future Improvements

- Add persistent caching for product-level Groq insights
- Store product-level sentiment outputs as JSON instead of generating them at runtime
- Improve category and luggage-size extraction with structured metadata
- Add downloadable reports and CSV exports from the dashboard
- Support multiple marketplaces beyond Amazon
- Add better testing and schema validation for processed data files

## License

This project is distributed under the terms of the [LICENSE](/c:/Users/Tanuj%20Rajput/OneDrive/Documents/GitHub/moonshot-dashboard/LICENSE) file in this repository.
