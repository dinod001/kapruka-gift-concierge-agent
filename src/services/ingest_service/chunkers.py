import json
from typing import Any, Dict, List


def get_product_chunks(json_file: str) -> List[Dict[str, Any]]:
    """Build custom product chunks from catalog JSON."""
    with open(json_file, "r", encoding="utf-8") as f:
        products = json.load(f)

    chunks: List[Dict[str, Any]] = []

    for chunk_id, p in enumerate(products, start=1):
        chunk_text = (
            f"Product: {p['product_name']}\n"
            f"Category: {p['category']}\n"
            f"Price: {p['price']}\n"
            f"Availability: {p['availability']}\n"
            f"Description: {p['description']}\n"
            f"Product URL: {p['product_url']}"
        )
        chunk = {
            "chunk_id": chunk_id,
            "product_name": p["product_name"],
            "category": p["category"],
            "price": p["price"],
            "availability": p["availability"],
            "description": p["description"],
            "product_url": p["product_url"],
            "chunk_text": chunk_text,
            "strategy": "custom",
        }

        chunks.append(chunk)

    return chunks