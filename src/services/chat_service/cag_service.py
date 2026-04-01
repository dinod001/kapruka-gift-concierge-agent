from loguru import logger
from typing import Any, Dict, List, Optional

from services.chat_service.cag_cache import CAGCache
from services.chat_service.crag_service import CRAGService


class CAGService:
    """
    Cache-Augmented Generation backed by Corrective RAG.

    Layer 1: Semantic cache (Qdrant cag_cache) -- instant, $0
    Layer 2: CRAG (confidence-gated retrieval) -- self-correcting
    """

    def __init__(self, crag_service: CRAGService, cache: CAGCache):
        self.crag_service = crag_service
        self.cache = cache


    def generate(
        self,
        query: str,
        use_cache: bool = True,
        memory_context: str = "",
        exclude_product: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate answer with CAG + CRAG pipeline.

        1. Check semantic cache (cosine >= 0.90 = HIT)
        2. On miss: run CRAGService (confidence-gated retrieval)
        3. Cache the result for future semantic hits
        """
        if use_cache:
            cached = self.cache.get(query)
            if cached:
                logger.info(
                    "CAG cache HIT (score={:.3f}) for: {}",
                    cached.get("score", 0),
                    query[:60],
                )
                return {
                    "answer": cached["answer"],
                    "product_url": cached.get("product_url", []),
                    "cache_hit": True,
                    "cache_score": cached.get("score", 0),
                    "generation_time": 0.0,
                }

        # Cache miss -- run CRAG (self-correcting retrieval)
        crag_result = self.crag_service.generate(
            query, 
            verbose=False,
            memory_context=memory_context,
            exclude_product=exclude_product
        )

        answer = crag_result.get("answer", "")
        product_url = crag_result.get("product_url", [])

        result: Dict[str, Any] = {
            "answer": answer,
            "product_url": product_url,
            "cache_hit": False,
            "confidence_initial": crag_result.get("confidence_initial", 0),
            "confidence_final": crag_result.get("confidence_final", 0),
            "correction_applied": crag_result.get("correction_applied", False),
            "generation_time": crag_result.get("generation_time", 0),
            "num_docs": crag_result.get("docs_used", 0),
        }

        if use_cache and answer:
            self.cache.set(query, {"answer": answer, "product_url": product_url})
            logger.info("CAG cache MISS -> cached for: {}", query[:60])

        return result

    def warm_cache(self, queries: List[str]) -> int:
        """Pre-populate cache with common queries via CRAG pipeline."""
        cached_count = 0
        for query in queries:
            if query not in self.cache:
                self.generate(query, use_cache=True)
                cached_count += 1
        return cached_count

    def cache_stats(self) -> Dict[str, Any]:
        return self.cache.stats()

    def clear_cache(self) -> None:
        self.cache.clear()


__all__ = ["CAGService"]
