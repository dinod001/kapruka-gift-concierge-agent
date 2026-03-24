#!/usr/bin/env python3
"""
CLI entry-point for the Qdrant ingestion pipeline.

All heavy-lifting lives in ``services.ingest_service.pipeline``.

Usage:
    PYTHONPATH=src python scripts/ingest_to_qdrant.py
    PYTHONPATH=src python scripts/ingest_to_qdrant.py --recreate
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# Ensure `src` is importable when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

load_dotenv()

from services.ingest_service.pipeline import run_ingest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest custom product chunks into Qdrant Cloud",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the Qdrant collection before ingesting",
    )
    args = parser.parse_args()

    run_ingest(
        source="catalog",
        strategy="custom",
        recreate=args.recreate,
    )


if __name__ == "__main__":
    main()
