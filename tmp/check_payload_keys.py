import os
import sys
from dotenv import load_dotenv

PROJECT_ROOT = r"E:\my career life\AI Enginert Essential\All Mini Projects\Mini project 03"
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_ROOT)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from infrastructure.db.qdrant_client import get_qdrant_client
from infrastructure.config import QDRANT_COLLECTION_NAME

client = get_qdrant_client()

print(f"Checking metadata keys for collection: {QDRANT_COLLECTION_NAME}")
try:
    results = client.scroll(
        collection_name=QDRANT_COLLECTION_NAME,
        limit=1,
        with_payload=True
    )[0]
    
    if results:
        payload = results[0].payload
        print(f"Payload Keys: {list(payload.keys())}")
        print(f"Full Payload: {payload}")
    else:
        print("No points found in collection.")
            
except Exception as e:
    print(f"Extraction failed: {e}")
