"""
Application configuration - loads from YAML param files.

CONFIGURATION POLICY:
====================
Configuration is loaded from config/param.yaml and config/models.yaml.
Secrets (API keys) live ONLY in .env and are loaded via os.getenv().

Supports multiple LLM providers via OpenRouter unified API or direct providers:
- OpenRouter (unified multi-provider access)
- OpenAI (direct)
- Anthropic (direct)
- Google/Gemini (direct)
- Groq (direct)
"""

from pathlib import Path
from typing import Any, Dict, Optional
import os
import yaml
from loguru import logger

# ========================================
# Project Paths
# ========================================

# Get project root (parent of src/infrastructure/)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"

# ========================================
# YAML Config Loading
# ========================================

def _load_yaml(filename: str) -> Dict[str, Any]:
    """Load a YAML config file."""
    filepath = _CONFIG_DIR / filename
    if not filepath.exists():
        return {}
    # On Windows, the default encoding may be cp1252 which can break on UTF-8
    # config files (e.g. emojis). Always read as UTF-8 with a BOM-safe fallback.
    try:
        text = filepath.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = filepath.read_text(encoding="utf-8-sig")
    return yaml.safe_load(text) or {}


def _get_nested(d: Dict, *keys, default=None):
    """Get nested dictionary value safely."""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d if d is not None else default


# Load configs
_PARAMS = _load_yaml("param.yaml")
_MODELS = _load_yaml("models.yaml")

# ========================================
# Provider Configuration
# ========================================

PROVIDER = _get_nested(_PARAMS, "provider", "default", default="openai")
MODEL_TIER = _get_nested(_PARAMS, "provider", "tier", default="general")
OPENROUTER_BASE_URL = _get_nested(_PARAMS, "provider", "openrouter_base_url",
                                   default="https://openrouter.ai/api/v1")

# ========================================
# Model Names (from models.yaml)
# ========================================

def get_chat_model(provider: Optional[str] = None, tier: Optional[str] = None) -> str:
    """Get chat model name for specified provider and tier."""
    provider = provider or PROVIDER
    tier = tier or MODEL_TIER

    # Handle provider name mapping
    if provider == "google":
        provider = "google"  # Keep as-is for models.yaml
    elif provider == "gemini":
        provider = "google"  # Alias

    return _get_nested(_MODELS, provider, "chat", tier, default="openai/gpt-4o-mini")


EMBEDDING_TIER = _get_nested(_PARAMS, "embedding", "tier", default="default")


def get_embedding_model(provider: Optional[str] = None, tier: Optional[str] = None) -> str:
    """Get embedding model name for specified provider and tier."""
    provider = provider or PROVIDER
    tier = tier or EMBEDDING_TIER

    # Handle provider name mapping
    if provider == "google" or provider == "gemini":
        provider = "google"

    return _get_nested(_MODELS, provider, "embedding", tier, default="openai/text-embedding-3-small")


# ========================================
# 3-Model Architecture
# ========================================
# Router/extractor can stay specialised, but the main chat model MUST match
# the active provider; otherwise you'll get "invalid model ID" from the API.

# ROUTER_MODEL = _get_nested(_MODELS, "openrouter", "chat", "general", default="openai/gpt-4o-mini")
# ROUTER_PROVIDER = "openrouter"

ROUTER_MODEL = _get_nested(_MODELS, "groq", "chat", "general", default="llama-3.1-8b-instant")
ROUTER_PROVIDER = "groq"

EXTRACTOR_MODEL = _get_nested(_MODELS, "groq", "chat", "general", default="llama-3.1-8b-instant")
EXTRACTOR_PROVIDER = "groq"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Main chat model/provider follow config/param.yaml provider settings.
CHAT_PROVIDER = PROVIDER
CHAT_MODEL = get_chat_model(provider=CHAT_PROVIDER, tier=MODEL_TIER)

EMBEDDING_MODEL = get_embedding_model()

# Legacy alias
OPENAI_CHAT_MODEL = CHAT_MODEL

# ========================================
# Embedding Dimensions
# ========================================

# ⚠️ IMPORTANT: EMBEDDING_DIM must match the model's output dimensions
#
# Supported models:
#   - "text-embedding-3-small"  → 1536 dims (recommended for Qdrant)
#   - "text-embedding-3-large"  → 3072 dims
#   - "text-embedding-ada-002"  → 1536 dims (legacy)
#
# Qdrant supports any dimension size (no limit like Supabase)
EMBEDDING_DIM = 1536  # Default for text-embedding-3-small

# Auto-detect dimension from model name
if "large" in EMBEDDING_MODEL.lower():
    EMBEDDING_DIM = 3072
elif "small" in EMBEDDING_MODEL.lower() or "ada" in EMBEDDING_MODEL.lower():
    EMBEDDING_DIM = 1536

# ========================================
# LLM Defaults
# ========================================

LLM_TEMPERATURE = _get_nested(_PARAMS, "llm", "temperature", default=0.0)
LLM_MAX_TOKENS = _get_nested(_PARAMS, "llm", "max_tokens", default=2000)
LLM_STREAMING = _get_nested(_PARAMS, "llm", "streaming", default=False)

# ========================================
# Embedding Defaults
# ========================================

EMBEDDING_BATCH_SIZE = _get_nested(_PARAMS, "embedding", "batch_size", default=100)
EMBEDDING_SHOW_PROGRESS = _get_nested(_PARAMS, "embedding", "show_progress", default=False)

# ========================================
# Project Paths (from param.yaml)
# ========================================

DATA_DIR = _PROJECT_ROOT / _get_nested(_PARAMS, "paths", "data_dir", default="data")
KB_DIR = _PROJECT_ROOT / _get_nested(_PARAMS, "paths", "kb_dir", default="data/knowledge_base")

# Active ingestion source: product catalog JSON file
JSONL_DIR = DATA_DIR / "jsonl"
MARKDOWN_DIR = _PROJECT_ROOT / _get_nested(_PARAMS, "paths", "markdown_dir", default="data/nawaloka_markdown")

# NOTE: Both RAG KB and CAG cache live in Qdrant Cloud (separate collections).
# NOTE: ST memory lives in Supabase (st_turns table).

# ========================================
# Chunking Configuration
# ========================================

CHUNKING_STRATEGY = _get_nested(_PARAMS, "chunking", "strategy", default="custom")

# ========================================
# Retrieval Configuration
# ========================================

TOP_K_RESULTS = _get_nested(_PARAMS, "retrieval", "top_k", default=8)
SIMILARITY_THRESHOLD = _get_nested(_PARAMS, "retrieval", "similarity_threshold", default=0.7)

# ========================================
# CAG Configuration (Qdrant Semantic Cache)
# ========================================

CAG_COLLECTION_NAME = _get_nested(_PARAMS, "cag", "collection_name", default="cag_cache")
CAG_SIMILARITY_THRESHOLD = _get_nested(_PARAMS, "cag", "similarity_threshold", default=0.90)
CAG_CACHE_TTL = _get_nested(_PARAMS, "cag", "cache_ttl", default=86400)  # 24h
CAG_CACHE_MAX_SIZE = _get_nested(_PARAMS, "cag", "max_cache_size", default=1000)

# ========================================
# CRAG Configuration
# ========================================

CRAG_CONFIDENCE_THRESHOLD = _get_nested(_PARAMS, "crag", "confidence_threshold", default=0.6)
CRAG_EXPANDED_K = _get_nested(_PARAMS, "crag", "expanded_k", default=12)

# ========================================
# Crawling Configuration
# ========================================

CRAWL_MAX_DEPTH = _get_nested(_PARAMS, "crawling", "max_depth", default=3)
CRAWL_DELAY_SECONDS = _get_nested(_PARAMS, "crawling", "delay_seconds", default=2.0)
CRAWL_MAX_PAGES = _get_nested(_PARAMS, "crawling", "max_pages", default=100)

# ========================================
# Memory Configuration (NOT in YAML - kept as constants)
# ========================================

# Timezone (used by web_search_tool for timestamp display)
TIMEZONE = "Asia/Colombo"

# Short-term memory (Supabase st_turns table)
ST_MAX_TURNS = 30
ST_TTL_SECONDS = 60 * 60 * 24  # 24 hours

# Long-term memory (Supabase Postgres + pgvector)
LT_TOP_K = 5
LT_SIM_THRESHOLD = 0.30  # Lowered from 0.65 for better recall
LT_TTL_SECONDS = 60 * 60 * 24 * 90  # 90 days
LT_DECAY_HALF_LIFE_DAYS = 30
MEM_COLLECTION = "mem_vectors"

# ========================================
# Reminders Configuration (FUTURE — not implemented in Week 07)
# Will be re-added when reminders_service is wired in.
# REM_TZ, REM_POLL_SECONDS, REM_DEFAULT_CHANNEL, QUIET_HOURS,
# REMINDER_OFFSETS_SECONDS, CRM_HORIZON_DAYS
# ========================================

# ========================================
# Database URLs
# ========================================

# Short-Term Memory: Supabase only (Redis introduced in a later week)

# ========================================
# Qdrant Cloud Configuration
# ========================================

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
QDRANT_URL = os.getenv("QDRANT_URL", None)
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "kapruka-gift-concierge-agent")

# ========================================
# FAQ Loading (optional)
# ========================================

def load_faqs() -> list:
    """
    Load known FAQs from config/faqs.yaml (if exists).

    Returns:
        List of FAQ question strings (flattened from all categories)
    """
    faqs_config = _load_yaml("faqs.yaml")
    if not faqs_config:
        return []

    all_faqs = []
    # Flatten all categories into a single list
    for category, questions in faqs_config.items():
        if isinstance(questions, list):
            all_faqs.extend(questions)

    return all_faqs


# Pre-load FAQs for easy access (empty if file doesn't exist)
KNOWN_FAQS = load_faqs()

# ========================================
# Helper Functions
# ========================================

def get_api_key(provider: Optional[str] = None) -> Optional[str]:
    """Get API key for the specified provider."""
    provider = provider or PROVIDER
    key_map = {
        "openrouter": "OPENROUTER_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "gemini": "GOOGLE_API_KEY",  # Alias
        "groq": "GROQ_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "tavily": "TAVILY_API_KEY",
    }
    env_var = key_map.get(provider, f"{provider.upper()}_API_KEY")
    return os.getenv(env_var)


def validate() -> None:
    """
    Validate configuration and create required directories.

    Raises:
        ValueError: If required secrets are missing
        OSError: If directories cannot be created
    """
    # Check required secrets based on provider
    api_key = get_api_key()
    if not api_key:
        key_name = "OPENROUTER_API_KEY" if PROVIDER == "openrouter" else f"{PROVIDER.upper()}_API_KEY"
        raise ValueError(
            f"❌ Missing required secret: {key_name}\n"
            f"Please add it to your .env file."
        )

    # Create required directories (only active ones)
    required_dirs = [DATA_DIR, KB_DIR.parent]

    for dir_path in required_dirs:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            raise OSError(f"❌ Cannot create directory {dir_path}: {e}")


def dump() -> None:
    """Print all active non-secret configuration values for debugging."""
    logger.info("\n" + "=" * 60)
    logger.info("CONFIGURATION (NON-SECRETS ONLY)")
    logger.info("=" * 60)

    logger.info("\n🌐 Provider:")
    logger.info(f"   Provider: {PROVIDER}")
    logger.info(f"   Model Tier: {MODEL_TIER}")
    logger.info(f"   Chat Model: {CHAT_MODEL}")
    logger.info(f"   Embedding Model: {EMBEDDING_MODEL}")
    logger.info(f"   Embedding Dimensions: {EMBEDDING_DIM}")

    logger.info("\n📁 Directories & Storage:")
    logger.info(f"   Data Root: {DATA_DIR}")
    logger.info(f"   Knowledge Base: {KB_DIR}")
    logger.info(f"   🟡 RAG Vectors: Qdrant Cloud ({QDRANT_COLLECTION_NAME})")
    logger.info(f"   🟡 CAG Cache: Qdrant Cloud ({CAG_COLLECTION_NAME})")
    logger.info(f"   🟢 ST + LT Memory + CRM: Supabase PostgreSQL")

    logger.info("\n🔧 Chunking:")
    logger.info(f"   Strategy: {CHUNKING_STRATEGY}")
    logger.info("   Mode: custom product chunks from catalog")

    logger.info("\n🔍 Retrieval:")
    logger.info(f"   Top-K Results: {TOP_K_RESULTS}")
    logger.info(f"   Similarity Threshold: {SIMILARITY_THRESHOLD}")

    logger.info("\n💾 CAG (Semantic Cache — Qdrant):")
    logger.info(f"   Collection: {CAG_COLLECTION_NAME}")
    logger.info(f"   Similarity Threshold: {CAG_SIMILARITY_THRESHOLD}")
    logger.info(f"   TTL (seconds): {CAG_CACHE_TTL}")
    logger.info(f"   Max Cache Size: {CAG_CACHE_MAX_SIZE}")

    logger.success("\n🎯 CRAG:")
    logger.info(f"   Confidence Threshold: {CRAG_CONFIDENCE_THRESHOLD}")
    logger.info(f"   Expanded K: {CRAG_EXPANDED_K}")

    logger.info("\n🧠 Memory:")
    logger.info(f"   Short-term Max Turns: {ST_MAX_TURNS}")
    logger.info(f"   Long-term Top-K: {LT_TOP_K}")
    logger.info(f"   Long-term Threshold: {LT_SIM_THRESHOLD}")

    logger.info("\n🗄️  Qdrant:")
    logger.info(f"   Collection: {QDRANT_COLLECTION_NAME}")
    logger.success(f"   URL: {'✅ Set' if QDRANT_URL else '❌ Not set'}")
    logger.success(f"   API Key: {'✅ Set' if QDRANT_API_KEY else '❌ Not set'}")

    logger.info("\n" + "=" * 60 + "\n")


def get_all_models() -> Dict[str, Any]:
    """Return all available models from models.yaml."""
    return _MODELS


def get_config() -> Dict[str, Any]:
    """Return full config dictionary."""
    return _PARAMS
