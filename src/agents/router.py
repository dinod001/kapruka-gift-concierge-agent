"""
Query Router — LLM-based intent classification.

Takes a user message + optional memory context, calls an LLM with a
structured JSON prompt, and returns a RouteDecision that tells the
orchestrator which specialist to invoke.

Valid routes:
  - "logistic" → delivery / district / shipping queries
  - "rag"      → product / gift / catalog / price queries
  - "direct"   → general conversation or fallback
"""

import json
from loguru import logger
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from agents.prompts.agent_prompts import build_router_prompt

# All routes the LLM is allowed to return
VALID_ROUTES = {"logistic", "rag", "direct"}


@dataclass
class RouteDecision:
    """
    Result of a single router LLM call.

    Attributes:
        route      : Selected specialist — "logistic" | "rag" | "direct".
        confidence : LLM's self-assessed confidence score (0.0 – 1.0).
        reasoning  : One-line explanation of why this route was chosen.
        action     : Optional sub-action (reserved for future specialists).
        params     : Entities extracted for the downstream agent
                     e.g. {"district": "Jaffna"} or {"query": "flower gift"}.
    """

    route: str = "direct"
    confidence: float = 0.0
    reasoning: str = ""
    action: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)


class QueryRouter:
    """
    Routes user queries to the appropriate specialist agent.

    Uses an LLM call with a strict JSON-output prompt to classify intent.
    Falls back to "direct" on any parse or LLM error.
    """

    def __init__(self, llm: Any) -> None:
        """
        Args:
            llm: A LangChain ChatOpenAI (or compatible) chat model instance.
                 Recommended: get_router_llm() from llm_provider.py.
        """
        self.llm = llm

    # ── public API ────────────────────────────────────────────────────────────

    def route(
        self,
        user_message: str,
        memory_context: str = "",
    ) -> RouteDecision:
        """
        Classify user intent and extract tool parameters.

        Builds a system + user prompt pair via build_router_prompt(), sends
        it to the LLM, and parses the JSON response into a RouteDecision.

        Args:
            user_message   : The raw user input string.
            memory_context : Optional stringified memory context that helps
                             the router understand what's already known.

        Returns:
            A RouteDecision. route is always one of VALID_ROUTES.
        """
        logger.info(f"[QueryRouter] Routing: '{user_message[:80]}'")

        system_prompt, user_prompt = build_router_prompt(
            user_message=user_message,
            memory_context=memory_context,
        )

        try:
            response = self.llm.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ]
            )
            # Normalise response to a plain string
            content = (
                response.content
                if hasattr(response, "content")
                else str(response)
            )

            # Log token usage if the provider exposes it
            if hasattr(response, "response_metadata"):
                meta = response.response_metadata or {}
                token_usage = meta.get("token_usage") or meta.get("usage", {})
                if token_usage:
                    logger.debug(
                        "[QueryRouter] Tokens — in:{} out:{} total:{}",
                        token_usage.get("prompt_tokens", 0),
                        token_usage.get("completion_tokens", 0),
                        token_usage.get("total_tokens", 0),
                    )

        except Exception as exc:
            logger.error("[QueryRouter] LLM call failed: {}", exc)
            return RouteDecision(
                route="direct",
                confidence=0.0,
                reasoning=f"Router LLM error: {exc}",
            )

        decision = self._parse_response(content)
        logger.info(
            "[QueryRouter] → route='{}' confidence={:.2f} model='{}' reason='{}'",
            decision.route,
            decision.confidence,
            self._model_name(),
            decision.reasoning,
        )
        return decision

    # ── private helpers ───────────────────────────────────────────────────────

    def _model_name(self) -> str:
        """Return the underlying LLM model name for logging."""
        if hasattr(self.llm, "model_name"):
            return self.llm.model_name
        if hasattr(self.llm, "model"):
            return self.llm.model
        return "unknown"

    def _parse_response(self, raw: str) -> RouteDecision:
        """
        Parse the JSON string returned by the router LLM.

        Handles:
          - Markdown code fences (```json ... ```)
          - Extra whitespace or preamble before the JSON object
          - Invalid JSON → falls back to route="direct"

        Args:
            raw: Raw LLM output string.

        Returns:
            A RouteDecision populated from the parsed JSON.
        """
        # Strip markdown fences if the LLM wrapped its output in them
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]   # drop the opening fence line
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]  # drop the closing fence
        text = text.strip()

        # Find the first complete JSON object in the output
        start = text.find("{")
        end   = text.rfind("}")
        if start == -1 or end == -1:
            logger.warning("[QueryRouter] Output is not JSON — falling back to direct.")
            return RouteDecision(
                route="direct",
                confidence=0.0,
                reasoning="Failed to parse router output as JSON.",
            )

        try:
            data = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            logger.warning("[QueryRouter] JSON parse error: {}", exc)
            return RouteDecision(
                route="direct",
                confidence=0.0,
                reasoning=f"JSON parse error: {exc}",
            )

        # Validate and sanitise route value
        route = data.get("route", "direct")
        if route not in VALID_ROUTES:
            logger.warning("[QueryRouter] Unknown route '{}' — falling back to direct.", route)
            route = "direct"

        return RouteDecision(
            route=route,
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", ""),
            params=data.get("params", {}),
        )
