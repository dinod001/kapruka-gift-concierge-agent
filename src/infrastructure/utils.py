"""
Helper functions for RAG pipeline.

Includes document formatting, confidence scoring, and citation utilities.
"""

import re
from typing import List


def calculate_confidence(docs: list, query: str) -> float:
    """
    Calculate confidence score for retrieved documents.

    Multi-factor heuristic:
    1. Keyword overlap (query ∩ docs)
    2. Content richness (avg doc length)
    3. Strategy diversity (historical; in this project we usually have 1 strategy)

    Args:
        docs: List of retrieved documents
        query: User query string

    Returns:
        Confidence score 0.0 to 1.0
    """
    if not docs:
        return 0.0

    # Extract query keywords
    query_words = set(query.lower().split())

    # Factor 1: Keyword overlap
    overlaps = []
    for doc in docs:
        doc_words = set(doc.page_content.lower().split())
        overlap = len(query_words & doc_words) / len(query_words) if query_words else 0
        overlaps.append(overlap)
    keyword_score = sum(overlaps) / len(overlaps)

    # Factor 2: Content richness
    avg_length = sum(len(doc.page_content) for doc in docs) / len(docs)
    length_score = min(avg_length / 500, 1.0)

    # Factor 3: Strategy diversity
    # In this mini project we typically run only a single chunking strategy ("custom"),
    # so we normalize diversity accordingly to keep confidence scale reasonable.
    strategies = set([doc.metadata.get('strategy', 'unknown') for doc in docs])
    if len(strategies) <= 1:
        diversity_score = 1.0
    else:
        # Backwards-compatible normalization for the old "up to 3 strategies" setup.
        diversity_score = len(strategies) / 3.0

    # Weighted average
    confidence = (
        0.5 * keyword_score +
        0.3 * length_score +
        0.2 * diversity_score
    )

    return confidence


def extract_citations(text: str) -> List[str]:
    """
    Extract [url] citations from generated text.

    Args:
        text: Generated answer with citations

    Returns:
        List of cited URLs
    """
    # Find all [content] patterns
    citations = re.findall(r'\[([^\]]+)\]', text)

    # Filter to actual URLs (contain http or .com)
    urls = [c for c in citations if 'http' in c or '.com' in c]

    return urls


def truncate_text(text: str, max_length: int = 400) -> str:
    """
    Truncate text to maximum length for quotes/previews.

    Args:
        text: Text to truncate
        max_length: Maximum character length

    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text

    return text[:max_length].rsplit(' ', 1)[0] + "..."
