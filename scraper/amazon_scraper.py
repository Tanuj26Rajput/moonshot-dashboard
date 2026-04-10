from playwright.sync_api import sync_playwright
import time
import random
import pandas as pd

data = []

def scrape_amazon_products(context):
    brands = ["Safari", "Nasher Miles", "Skybags", "uppercase", "FUR JADEN"]
    page = context.new_page()
    for brand in brands:
        page.goto(f"https://www.amazon.in/s?k={brand.lower()}+luggage")
        time.sleep(3)
        products = page.query_selector_all("div.s-main-slot div[data-component-type='s-search-result']")
        for product in products[:12]:
            spans = product.query_selector_all("h2 span")
            # Title
            try:
                if(len(spans) > 1):
                    if(brand == spans[0].inner_text()):
                        title = spans[1].inner_text()
                else:
                    title = spans[0].inner_text()
            except:
                title = None
            # Price
            try:
                if(brand == spans[0].inner_text()):
                    price = product.query_selector(".a-price-whole").inner_text()
            except:
                price = None
            # Rating
            try:
                if(brand == spans[0].inner_text()):
                    rating = product.query_selector(".a-icon-alt").inner_text()
            except:
                rating = None
            try:
                if(brand == spans[0].inner_text()):
                    asin = product.get_attribute("data-asin")
                    url = f"https://www.amazon.in/dp/{asin}"
            except:
                asin = None
                url = None
            data.append({
                "brand": brand,
                "asin": asin,
                "title": title,
                "price": price,
                "rating": rating,
                "url": url
            })
    return data
    
def setup_logged_in_context():
    from playwright.sync_api import sync_playwright

    p = sync_playwright().start()

    context = p.chromium.launch_persistent_context(
        user_data_dir="user_data",
        headless=False
    )

    page = context.new_page()

    page.goto("https://www.amazon.in")

    input("👉 Login manually, then press ENTER here...")

    return p, context

def scrape_reviews(asin, context, max_reviews=60):
    page = context.new_page()

    url = f"https://www.amazon.in/product-reviews/{asin}"
    page.goto(url, timeout=60000)

    reviews_data = []
    seen = set()

    while len(reviews_data) < max_reviews:

        reviews = page.query_selector_all("[data-hook='review']")
        for r in reviews:
            if len(reviews_data) >= max_reviews:
                break

            try:
                rating = r.query_selector(
                    "i[data-hook='review-star-rating'] span"
                ).inner_text()
            except:
                rating = None

            try:
                title = r.query_selector(
                    "a[data-hook='review-title'] span:not(.a-icon-alt)"
                ).inner_text()
            except:
                title = None

            try:
                body = r.query_selector(
                    "span[data-hook='review-body'] span"
                ).inner_text()
            except:
                body = None

            if not body or body in seen:
                continue
            
            seen.add(body)
            
            reviews_data.append({
                "rating": rating,
                "title": title,
                "body": body
            })
        
        time.sleep(random.uniform(2, 4))
        page.mouse.wheel(0, 3000)

        try:
            show_more = page.query_selector("a[data-hook='show-more-button']")

            if show_more:
                show_more.scroll_into_view_if_needed()
                show_more.click()
                page.wait_for_timeout(3000)
            else:
                break
        except:
            break

    return reviews_data

def convert_to_dataframe(products):
    rows = []

    for product in products:
        for review in product.get('reviews', []):
            rows.append({
                "brand": product.get('brand'),
                "asin": product.get('asin'),
                "product_title": product.get('title'),
                "price": product.get('price'),
                "product_rating": product.get('rating'),
                "review_rating": review.get('rating'),
                "review_title": review.get('title'),
                'review_body': review.get('body'),
                "url": product.get('url'),
            })
    
    df = pd.DataFrame(rows)
    return df

            
if __name__ == "__main__":
    p, context = setup_logged_in_context()
    products = scrape_amazon_products(context)

    for product in products:
        product['reviews'] = scrape_reviews(product['asin'], context)
    
    df = convert_to_dataframe(products)

    df.to_csv("data/raw/amazon_luggage.csv", index=False)
    print("\nData saved to amazon_luggage.csv")

    context.close()
    p.stop()