"""
CRAG (Corrective RAG) service with self-correcting retrieval.

Provides:
- CRAGService: Self-correcting RAG with confidence scoring
- Automatic query expansion on low confidence
- Better grounding and reduced hallucinations

Workflow:
    1. Initial retrieval (k=4)
    2. Calculate confidence score
    3. If low: Corrective retrieval (k=8, expanded)
    4. Generate with best evidence

Benefits:
- 🎯 Better accuracy for complex queries
- 🛡️ Reduces hallucinations
- 🔄 Automatic self-correction
"""

from loguru import logger
from typing import Any, Dict, List, Optional
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever

from infrastructure.config import (
    CRAG_CONFIDENCE_THRESHOLD,
    CRAG_EXPANDED_K,
    TOP_K_RESULTS
)
from services.chat_service.rag_templates import RAG_TEMPLATE
from services.chat_service.rag_service import QdrantRetriever
from infrastructure.utils import calculate_confidence


class CRAGService:
    """
    Corrective RAG service with automatic self-correction.
    
    Features:
    - Initial retrieval with confidence scoring
    - Automatic corrective retrieval if confidence low
    - Query expansion strategies
    - Detailed metrics for debugging
    
    Usage:
        service = CRAGService(retriever, llm)
        result = service.generate(query, confidence_threshold=0.6)
        
        logger.info(result['answer'])
        logger.info(f"Confidence: {result['confidence_final']}")
        logger.info(f"Correction applied: {result['correction_applied']}")
    """
    
    def __init__(
        self,
        retriever: BaseRetriever,
        llm: Any,
        initial_k: int = TOP_K_RESULTS,
        expanded_k: int = CRAG_EXPANDED_K
    ):
        """
        Initialize CRAG service.
        
        Args:
            retriever: LangChain retriever (QdrantRetriever, etc.)
            llm: LangChain LLM instance
            initial_k: Number of docs for initial retrieval
            expanded_k: Number of docs for corrective retrieval
        """
        self.retriever = retriever
        self.llm = llm
        self.initial_k = initial_k
        self.expanded_k = expanded_k
        
        # Create prompt
        self.prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

    def _set_k(self, k: int) -> None:
        """Set retrieval count on the retriever (supports both patterns)."""
        if isinstance(self.retriever, QdrantRetriever):
            self.retriever.top_k = k
        elif hasattr(self.retriever, "search_kwargs"):
            self.retriever.search_kwargs["k"] = k
    
    def generate(
        self,
        query: str,
        confidence_threshold: float = CRAG_CONFIDENCE_THRESHOLD,
        verbose: bool = True,
        memory_context: str = "",
        exclude_product: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate answer with CRAG (Corrective RAG).
        
        Workflow:
        1. Initial retrieval (k=initial_k)
        2. Calculate confidence score
        3. If confidence < threshold:
           - Apply corrective retrieval (k=expanded_k)
           - Recalculate confidence
        4. Generate answer with best evidence
        
        Args:
            query: User question
            confidence_threshold: Minimum confidence score (0-1)
            verbose: Print progress logs
        
        Returns:
            Dict with:
            - answer: Generated answer
            - confidence_initial: Initial confidence score
            - confidence_final: Final confidence score
            - correction_applied: Whether corrective retrieval was used
            - docs_used: Number of documents in final generation
            - generation_time: Total time taken
            - evidence_urls: List of source URLs
        """
        if verbose:
            logger.info(f"🔍 Query: {query}")
            logger.success(f"🎯 Confidence threshold: {confidence_threshold}\n")
        
        # --- Step 1: Initial retrieval ---
        # If we have an exclusion, use a larger initial K to find alternatives
        effective_initial_k = self.initial_k
        if exclude_product:
            effective_initial_k = max(self.initial_k, 15)
            if verbose:
                logger.info(f"🔍 Exclusion detected: '{exclude_product}'. Using deep search (k={effective_initial_k})...")
        
        self._set_k(effective_initial_k)
        docs_initial = self.retriever.invoke(query)
        confidence_initial = calculate_confidence(docs_initial, query)
        
        if verbose:
            logger.info(f"   📊 Confidence: {confidence_initial:.2f}")
        
        # Step 2: Check if correction needed
        if confidence_initial >= confidence_threshold:
            if verbose:
                logger.success(f"   ✅ Confidence sufficient - proceeding with initial retrieval")
            final_docs = docs_initial
            confidence_final = confidence_initial
            correction_applied = False
        else:
            if verbose:
                logger.warning(f"   ⚠️  Low confidence - applying corrective retrieval...\n")
            
            # Step 3: Corrective retrieval
            # If we have an exclusion, push expansion even further
            effective_expanded_k = self.expanded_k
            if exclude_product:
                effective_expanded_k = max(self.expanded_k, 20)

            if verbose:
                logger.info(f"2️⃣  Corrective retrieval (k={effective_expanded_k}, expanded)...")
            
            self._set_k(effective_expanded_k)
            docs_corrected = self.retriever.invoke(query)
            confidence_final = calculate_confidence(docs_corrected, query)
            
            if verbose:
                logger.info(f"   📊 Corrected confidence: {confidence_final:.2f}")
                improvement = (confidence_final - confidence_initial) * 100
                if verbose:
                    logger.info(f"   📈 Confidence improved by {improvement:.1f}%")
            
            final_docs = docs_corrected
            correction_applied = True
        
        # --- NEW Filtering step: Remove excluded product (Negative constraint) ---
        if exclude_product:
            original_count = len(final_docs)
            excl_lower = exclude_product.lower().strip()
            # Filter matches by checking product_name in metadata (bi-directional)
            # This handles cases like 'Nubia Neo 3' vs 'Nubia Neo 3 5G 8GB Shadow Black'
            # Also check page_content for robustness
            final_docs = [
                doc for doc in final_docs 
                if (
                    (doc.metadata.get("product_name", "").lower() not in excl_lower) and 
                    (excl_lower not in doc.metadata.get("product_name", "").lower()) and
                    (excl_lower not in doc.page_content.lower())
                )
            ]
            if verbose:
                removed = original_count - len(final_docs)
                logger.success(f"🚫 Negative Filter: Removed {removed} chunks matching '{exclude_product}'")
        # -------------------------------------------------------------------------
        
        # --- Diversity Step: Ensure unique products are represented ---
        # Group docs by product_name and keep only the best 2 chunks per product
        # to prevent one product from swallowing the full Top-K context.
        unique_products = {}
        for doc in final_docs:
            p_name = doc.metadata.get("product_name", "Unknown")
            if p_name not in unique_products:
                unique_products[p_name] = []
            
            # Limit to 2 most relevant chunks per product
            if len(unique_products[p_name]) < 2:
                unique_products[p_name].append(doc)
        
        # Re-flatten back into a list of docs (preserving relevance order across different products)
        diverse_docs = []
        for p_name in unique_products:
            diverse_docs.extend(unique_products[p_name])
            
        final_docs = diverse_docs
        # -------------------------------------------------------------
        
        # Step 4: Generate answer
        if verbose:
            logger.info(f"\n3️⃣  Generating answer...")
        
        start = time.time()
        
        # Format docs and generate:
        context_text = "\n\n".join(doc.page_content for doc in final_docs)
        
        # --- NEW: Empty Context Guard (Anti-Hallucination) ---
        # If filtering removed everything, explicitly tell the LLM NOT to recommend the excluded product.
        if not context_text.strip() and exclude_product:
            context_text = (
                f"TECHNICAL NOTE: The user specifically asked for an alternative to '{exclude_product}'. "
                f"However, NO OTHER DIFFERENT PRODUCTS were found in the catalog search for their category. "
                f"STRICT RULE: Do NOT suggest '{exclude_product}' or any of its variations. "
                "Inform the user that we currently don't have other matching options available."
            )

        # Combine memory context (history) with RAG evidence for better grounding
        full_context = context_text
        if memory_context:
            full_context = f"--- RECENT CONVERSATION ---\n{memory_context}\n\n--- PRODUCT EVIDENCE ---\n{context_text}"
            
        prompt_input = {"context": full_context, "question": query}
        answer = (self.prompt | self.llm | StrOutputParser()).invoke(prompt_input)
        
        elapsed = time.time() - start
        
        # Extract evidence URLs (ignore empty/missing)
        product_url = list(
            {
                doc.metadata.get("product_url", "")
                for doc in final_docs
                if doc.metadata.get("product_url", "")
            }
        )
        
        return {
            'answer': answer,
            'confidence_initial': confidence_initial,
            'confidence_final': confidence_final,
            'correction_applied': correction_applied,
            'docs_used': len(final_docs),
            'generation_time': elapsed,
            'product_url': product_url,
            'evidence': final_docs
        }
    
    def batch_generate(
        self,
        queries: List[str],
        confidence_threshold: float = CRAG_CONFIDENCE_THRESHOLD
    ) -> List[Dict[str, Any]]:
        """
        Generate answers for multiple queries with CRAG.
        
        Args:
            queries: List of user questions
            confidence_threshold: Minimum confidence score
        
        Returns:
            List of result dicts (same format as generate())
        """
        results = []
        for query in queries:
            result = self.generate(query, confidence_threshold, verbose=False)
            results.append(result)
        return results
    
    def analyze_confidence(self, query: str) -> Dict[str, Any]:
        """
        Analyze confidence without generating answer (for debugging).
        
        Args:
            query: User question
        
        Returns:
            Dict with confidence metrics
        """
        # Initial retrieval
        self._set_k(self.initial_k)
        docs_initial = self.retriever.invoke(query)
        confidence_initial = calculate_confidence(docs_initial, query)
        
        # Expanded retrieval
        self._set_k(self.expanded_k)
        docs_expanded = self.retriever.invoke(query)
        confidence_expanded = calculate_confidence(docs_expanded, query)
        
        return {
            'query': query,
            'confidence_initial': confidence_initial,
            'confidence_expanded': confidence_expanded,
            'improvement': confidence_expanded - confidence_initial,
            'docs_initial': len(docs_initial),
            'docs_expanded': len(docs_expanded)
        }


__all__ = ['CRAGService']

