"""
Kapruka Product Scraper  –  python web_crawler.py
Edit the CATEGORIES list at the bottom and run.
"""

import asyncio
import json
from pathlib import Path

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

INTERNATIONAL = {"usa", "canada", "australia", "uk", "uae", "france", "new_zealand", "india", "exports"}


# ── helpers ──────────────────────────────────────────────────────────────────

def _category_name(url: str) -> str:
    if "srilanka_online_catalogue.jsp" in url:
        import urllib.parse
        buy = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("buy", ["unknown"])[0]
        return f"{buy} (Food)"
    slug = url.rstrip("/").split("/")[-1]
    if "/food/" in url:
        return f"{slug} (Food)"
    if slug.lower() in INTERNATIONAL:
        return f"{slug} (International)"
    return slug


def _is_product_href(h: str) -> bool:
    return bool(h and ("/buyonline/" in h or "/buyinternational/" in h))


async def _handle_age_modal(page):
    """Click the 21+ age verification button if it appears."""
    try:
        btn = await page.query_selector("#age-yes")
        if btn and await btn.is_visible():
            await btn.click()
            await page.wait_for_timeout(1000)
    except Exception:
        pass


# ── step 1: collect product links ────────────────────────────────────────────

async def collect_product_links(page, category_url: str, max_products: int = 100) -> list[str]:
    print(f"\n  Opening: {category_url}")
    await page.goto(category_url, wait_until="domcontentloaded", timeout=30000)
    await page.wait_for_timeout(2500)
    await _handle_age_modal(page)

    click_count = 0
    while True:
        html = await page.content()
        soup = BeautifulSoup(html, "html.parser")
        visible = len(soup.find_all("a", href=_is_product_href))
        print(f"  Click {click_count:>3} → {visible} products visible…", end="\r")

        if visible >= max_products:
            print(f"\n  Reached limit ({max_products}) — stopping.")
            break

        btn = await page.query_selector("#viewMoreButton")
        if btn is None or not await btn.is_visible():
            print(f"\n  All products loaded after {click_count} clicks.")
            break

        await btn.scroll_into_view_if_needed()
        await btn.click()
        click_count += 1
        await page.wait_for_timeout(2000)

    html = await page.content()
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        if _is_product_href(a["href"]):
            href = a["href"]
            full = href if href.startswith("http") else "https://www.kapruka.com" + href
            links.add(full.split("?")[0].split("#")[0])
            if len(links) >= max_products:
                break

    print(f"\n  Found {len(links)} unique product URLs.")
    return list(links)


# ── step 2: extract details from a product page ───────────────────────────────

def extract_product_details(soup: BeautifulSoup, url: str, category: str) -> dict:
    # Name
    name_block = soup.find("div", class_=lambda c: c and "blockDelivery" in c and "imgtags" in c)
    h1 = name_block.find("h1") if name_block else soup.find("h1")
    name = h1.get_text(strip=True) if h1 else "N/A"

    # Price (discounted → standard → alt layout → priceM class)
    price_tag = (
        soup.find(id="priceAfterDiscountlbl") or
        soup.find(id="pricelbl") or
        soup.select_one("div.price.priceMobileFix strong") or
        soup.select_one("div.price strong") or
        soup.find(class_="priceM")
    )
    price = price_tag.get_text(strip=True) if price_tag else "N/A"

    # Description (standard → international layout)
    desc_div = soup.find("div", class_="detailDescription") or soup.find("div", class_="info-wrap")
    description = desc_div.get_text(separator=" ", strip=True) if desc_div else ""

    # Availability
    availability = "Unknown"
    avail_tag = soup.select_one("div.tagArea span.tags")
    if avail_tag:
        t = avail_tag.get_text().lower()
        if "in stock" in t:      availability = "In Stock"
        elif "out of stock" in t: availability = "Out of Stock"
        elif "pre-order" in t:    availability = "Pre-order"
        else:                     availability = t.strip().title()

    if availability in ("Unknown", ""):
        btn = soup.find("button", class_=lambda c: c and "cart" in c)
        if btn:
            t = btn.get_text().lower()
            if "out of stock" in t:              availability = "Out of Stock"
            elif "in stock" in t or "add" in t:  availability = "In Stock"

    if availability in ("Unknown", ""):
        t = soup.get_text().lower()
        if "out of stock" in t:  availability = "Out of Stock"
        elif "in stock" in t:    availability = "In Stock"

    return {
        "product_name": name,
        "price":        price,
        "description":  description,
        "availability": availability,
        "category":     category,
        "product_url":  url,
    }


# ── step 3: visit a product page (with retry) ────────────────────────────────

async def scrape_product(page, url: str, category: str, delay: float = 1.5) -> dict:
    for attempt in range(1, 4):
        try:
            if attempt > 1:
                wait = attempt * 3.0
                print(f"    ↩ Retry {attempt - 1} (waiting {wait:.0f}s)…")
                await asyncio.sleep(wait)

            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await _handle_age_modal(page)

            try:
                await page.wait_for_selector("div.blockDelivery.imgtags h1, span#pricelbl", timeout=10000)
            except Exception:
                await page.wait_for_timeout(2000)

            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(500)

            soup = BeautifulSoup(await page.content(), "html.parser")
            product = extract_product_details(soup, url, category)

            bad = product["product_name"] in ("www.kapruka.com", "N/A", "") or product["price"] == "N/A"
            if bad and attempt < 3:
                print(f"    ⚠ Bad data on attempt {attempt} – retrying…")
                continue

            return product

        except Exception as err:
            if attempt < 3:
                print(f"    ✗ Error: {str(err)[:80]} – retrying…")
            else:
                return {
                    "product_name": "ERROR",
                    "price":        "N/A",
                    "description":  "N/A",
                    "availability": "N/A",
                    "category":     category,
                    "product_url":  url,
                    "error":        str(err)[:200],
                }


# ── step 4: scrape all categories ────────────────────────────────────────────

async def scrape_categories(category_urls: list[str], delay: float = 1.5, max_products: int = 100) -> list[dict]:
    all_products = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.set_extra_http_headers({"User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )})

        for cat_url in category_urls:
            category = _category_name(cat_url)
            print(f"\n{'═' * 50}\nCategory: {category.upper()}\n{'═' * 50}")

            product_urls = await collect_product_links(page, cat_url, max_products)
            to_scrape = product_urls[:max_products]
            print(f"  Scraping {len(to_scrape)} product pages…")

            for i, url in enumerate(to_scrape, 1):
                print(f"  [{i:>4}/{len(product_urls)}] ", end="")
                product = await scrape_product(page, url, category, delay)
                all_products.append(product)
                status = "✓" if product.get("price") != "N/A" else "✗"
                print(f"{status} {product['product_name']}  |  {product['price']}")
                await asyncio.sleep(delay)

        await browser.close()

    return all_products


# ── step 5: save to JSON (appends, no duplicates) ────────────────────────────

def run(category_urls: list[str], save_json: str | None = None, delay: float = 1.5,
        max_products: int = 100, clear_old: bool = False) -> list[dict]:
    new_products = asyncio.run(scrape_categories(category_urls, delay, max_products))

    if save_json:
        path = Path(save_json)
        final_list = []

        if not clear_old and path.exists():
            try:
                final_list = json.load(open(path, encoding="utf-8"))
                if not isinstance(final_list, list):
                    final_list = []
            except Exception:
                final_list = []

        existing_urls = {p.get("product_url") for p in final_list}
        added = 0
        for p in new_products:
            if p.get("product_url") not in existing_urls:
                final_list.append(p)
                added += 1

        json.dump(final_list, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        label = "replaced" if clear_old else f"added {added} new"
        print(f"\n✓ {label.capitalize()} products (Total: {len(final_list)}) → {save_json}")

    return new_products


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    CATEGORIES = [
        # ── Local ─────────────────────────────────────────────────────────────
        # "https://www.kapruka.com/online/cakes",
        # "https://www.kapruka.com/online/chocolates",
        # "https://www.kapruka.com/online/combogifts",
        # "https://www.kapruka.com/online/clothing",
        #"https://www.kapruka.com/online/electronics",
        # "https://www.kapruka.com/online/flowers",
        # "https://www.kapruka.com/online/fruitbaskets",
        # "https://www.kapruka.com/online/vegetables",
        # "https://www.kapruka.com/online/giftvouchers",
        #"https://www.kapruka.com/online/giftset",
        #"https://www.kapruka.com/online/grocery",
        # "https://www.kapruka.com/online/greetingcards",
        # "https://www.kapruka.com/online/hampers",
        # "https://www.kapruka.com/online/Jewellery",
        # "https://www.kapruka.com/online/customizedGifts",
        # "https://www.kapruka.com/online/perfumes",
        # "https://www.kapruka.com/online/fashion",
        # "https://www.kapruka.com/online/cosmetics",
        # "https://www.kapruka.com/online/schoolpride",
        # "https://www.kapruka.com/online/childrens",
        # "https://www.kapruka.com/online/books",
        # "https://www.kapruka.com/online/pharmacy",
        # "https://www.kapruka.com/online/easter",
        # "https://www.kapruka.com/online/sports",
        # "https://www.kapruka.com/online/baby",
        #"https://www.kapruka.com/online/home_lifestyle",
        # "https://www.kapruka.com/online/pirikara",
        #"https://www.kapruka.com/online/Automobile",
        #"https://www.kapruka.com/online/Intimate_Essentials",
        # "https://www.kapruka.com/online/pet",

        # ── International ──────────────────────────────────────────────────────
        # "https://www.kapruka.com/online/exports",
        # "https://www.kapruka.com/online/USA",
        # "https://www.kapruka.com/online/Canada",
        # "https://www.kapruka.com/online/Australia",
        # "https://www.kapruka.com/online/UK",
        # "https://www.kapruka.com/online/UAE",
        # "https://www.kapruka.com/online/France",
        # "https://www.kapruka.com/online/new_zealand",
        # "https://www.kapruka.com/online/India",

        # ── Food / Restaurants ────────────────────────────────────────────────
        # "https://www.kapruka.com/srilanka_online_catalogue.jsp?buy=Curd",
        # "https://www.kapruka.com/food/happy_panda",
        # "https://www.kapruka.com/food/jasmine_song",
        # "https://www.kapruka.com/food/sizzle",
        # "https://www.kapruka.com/food/grand_chancellor_kitchens",
        # "https://www.kapruka.com/food/manhattan",
        # "https://www.kapruka.com/food/chinese_dragon",
        # "https://www.kapruka.com/food/crystal_jade",
        # "https://www.kapruka.com/food/mama_ranas",
        # "https://www.kapruka.com/food/shanmugas",
        # "https://www.kapruka.com/food/hot_kitchen",
        # "https://www.kapruka.com/food/mirchiz",
        # "https://www.kapruka.com/food/indian_summer",
        # "https://www.kapruka.com/food/dominos",
        # "https://www.kapruka.com/food/pizzahut",
        # "https://www.kapruka.com/food/mitsis",
        # "https://www.kapruka.com/food/arabian_vibes",
        # "https://www.kapruka.com/food/cinnamon_lakeside",
        # "https://www.kapruka.com/food/kingsbury",
        # "https://www.kapruka.com/food/galadari",
        # "https://www.kapruka.com/food/cinnamon_grand",
        # "https://www.kapruka.com/food/raja_bojun",
        # "https://www.kapruka.com/food/kottu_lab",
        # "https://www.kapruka.com/food/mr_kottu_grand",
        # "https://www.kapruka.com/food/dinemore",
        # "https://www.kapruka.com/food/subway",
        # "https://www.kapruka.com/food/popeyes",
        # "https://www.kapruka.com/food/taco_bell",
        # "https://www.kapruka.com/food/divine",
        # "https://www.kapruka.com/food/delifrance",
        # "https://www.kapruka.com/food/green_cabin",
        # "https://www.kapruka.com/food/paanpaan",
    ]

    OUTPUT_FILE = Path(__file__).parent.parent.parent.parent / "data" / "Catalog.json"
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    products = run(
        category_urls=CATEGORIES,
        save_json=str(OUTPUT_FILE),
        delay=1.5,
        max_products=100,
    )

    print(f"\n{'─' * 50}")
    print(f"Total products scraped: {len(products)}")
    if products:
        print(json.dumps(products[0], ensure_ascii=False, indent=2))