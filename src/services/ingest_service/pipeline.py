"""Qdrant ingestion pipeline for custom product chunking."""

from loguru import logger
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from infrastructure.config import KB_DIR, QDRANT_COLLECTION_NAME, EMBEDDING_BATCH_SIZE
from infrastructure.llm.embeddings import get_default_embeddings
from infrastructure.db.qdrant_client import (
    ensure_collection,
    delete_collection,
    upsert_chunks,
    collection_info,
)
from services.ingest_service.chunkers import get_product_chunks


# =====================================================================
# Document loaders
# =====================================================================


def load_catalog_chunks(catalog_file: Path | None = None) -> List[Dict[str, Any]]:
    """Load product chunks from the configured catalog JSON file."""
    catalog_file = Path(catalog_file or KB_DIR)
    if not catalog_file.exists():
        raise FileNotFoundError(f"Catalog file not found: {catalog_file}")

    chunks = get_product_chunks(str(catalog_file))
    logger.info("Loaded {} product chunks from {}", len(chunks), catalog_file)
    return chunks


# =====================================================================
# Embedding helper
# =====================================================================


def embed_texts(
    texts: List[str],
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> List[List[float]]:
    """Embed a list of texts using the configured embedding model."""
    embedder = get_default_embeddings(batch_size=batch_size)
    all_embeddings: List[List[float]] = []

    total_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_num = i // batch_size + 1
        logger.info(
            "Embedding batch {}/{} ({} texts)...",
            batch_num,
            total_batches,
            len(batch),
        )
        batch_embeddings = embedder.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def run_ingest(
    source: str = "catalog",
    strategy: str = "custom",
    recreate: bool = False,
) -> int:
    """
    End-to-end ingestion pipeline.

    Args:
        source: Only ``catalog`` is supported.
        strategy: Only ``custom`` is supported.
        recreate: If ``True``, drop and recreate the Qdrant collection first.

    Returns:
        Number of points upserted.

    Raises:
        ValueError: If *source* or *strategy* is not ``catalog``/``custom``.
        FileNotFoundError: If the catalog JSON file does not exist.
    """
    logger.info("=" * 70)
    logger.info("🚀 QDRANT INGESTION PIPELINE")
    logger.info("=" * 70)

    if source != "catalog":
        raise ValueError("Only source='catalog' is supported for this project.")
    if strategy != "custom":
        raise ValueError("Only strategy='custom' is supported for this project.")

    # ── 1. Load custom chunks ────────────────────────────────
    logger.info("\n📂 Loading product catalog chunks...")
    chunks = load_catalog_chunks()
    if not chunks:
        logger.error("❌ No chunks loaded from catalog. Nothing to ingest.")
        sys.exit(1)

    # ── 2. Embed ─────────────────────────────────────────────
    logger.info(f"\n🔢 Embedding {len(chunks)} chunks...")
    texts = [c["chunk_text"] for c in chunks]
    t0 = time.time()
    embeddings = embed_texts(texts)
    embed_secs = time.time() - t0
    logger.success(f"   → Embedding done in {embed_secs:.1f}s")

    # ── 3. Create / recreate collection ──────────────────────
    if recreate:
        logger.info(f"\n🗑️  Recreating collection '{QDRANT_COLLECTION_NAME}'...")
        try:
            delete_collection()
        except Exception:
            pass  # collection may not exist yet

    ensure_collection()

    # ── 4. Upsert ────────────────────────────────────────────
    logger.info(f"\n⬆️  Upserting {len(chunks)} points into Qdrant...")
    t0 = time.time()
    n = upsert_chunks(chunks, embeddings)
    upsert_secs = time.time() - t0
    logger.info(f"   → Upserted {n} points in {upsert_secs:.1f}s")

    # ── 5. Verify ────────────────────────────────────────────
    logger.info("\n📊 Collection info:")
    info = collection_info()
    for k, v in info.items():
        logger.info(f"   {k}: {v}")

    logger.info("\n" + "=" * 70)
    logger.success("✅ INGESTION COMPLETE")
    logger.info(f"   Source: {source}")
    logger.info(f"   Strategy: {strategy}")
    logger.info(f"   Chunks indexed: {n}")
    logger.info("=" * 70)

    return n
