import os
import sys
from dotenv import load_dotenv

PROJECT_ROOT = r"E:\my career life\AI Enginert Essential\All Mini Projects\Mini project 03"
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_ROOT)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from infrastructure.db.qdrant_client import search_chunks
from services.ingest_service.embeddings import get_default_embeddings

embedder = get_default_embeddings()

def check_item(query):
    print(f"\nSearching for '{query}' with TOP_K=20...")
    try:
        query_vec = embedder.embed_query(query)
        # Search with a larger K to see all phones
        results = search_chunks(query_vector=query_vec, top_k=20)
        if not results:
            print("No results found.")
            return
        
        seen_products = {}
        for r in results:
            name = r.get("product_name", "N/A")
            score = r.get("score", 0)
            if name not in seen_products:
                seen_products[name] = score
        
        print(f"Found {len(seen_products)} unique products:")
        for name, score in seen_products.items():
            print(f"- {name} (Top score: {score:.2f})")
            
    except Exception as e:
        print(f"Search failed: {e}")

check_item("smartphone")
