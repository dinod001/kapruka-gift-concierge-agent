"""
Agent Orchestrator — main execution loop.

Flow:
  1. Recall memory context (ST chat history + LT profiles).
  2. Route the user query (LLM → RouteDecision).
  3. Dispatch to the selected tool (logistic | rag | direct).
  4. Synthesise initial draft answer using the chat LLM.
  5. Reflect  — check draft against recipient profiles for violations.
  6. Revise   — if violation found, fix the draft (else return as-is).
  7. Save the final turn to memory (ST always, LT if intent triggered).
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from loguru import logger

from agents.router import QueryRouter, RouteDecision
from agents.prompts.agent_prompts import build_reflect_prompt, build_revise_prompt
from memory.memory_ops import MemoryDistiller


@dataclass
class AgentResponse:
    """
    Complete agent response with metadata.

    Attributes:
        answer     : The final (post-reflection) text response shown to the user.
        route      : Which route was selected (logistic | rag | direct).
        tool_output: Raw tool output before synthesis (for debugging).
        violated   : True if a reflection violation was found and revised.
        latency_ms : End-to-end processing time in milliseconds.
    """

    answer: str
    route: str = "direct"
    tool_output: str = ""
    violated: bool = False
    latency_ms: int = 0


class AgentOrchestrator:
    """
    Main agent loop that ties routing, tools, memory, and synthesis together.

    Pipeline: Recall → Route → Dispatch → Draft → Reflect → Revise → Save

    Dependencies (injected via __init__):
        llm_chat      — Chat LLM for synthesis/reflect/revise (e.g. Gemini 2.0 Flash)
        llm_router    — Router LLM for intent classification (e.g. GPT-4o-mini)
        memory        — MemoryDistiller (ST + LT memory coordinator)
        logistic_tool — LogisticAlert agent (optional)
        rag_tool      — RAGTool (optional)
    """

    def __init__(
        self,
        llm_chat: Any,
        llm_router: Any,
        memory: MemoryDistiller,
        logistic_tool: Optional[Any] = None,
        rag_tool: Optional[Any] = None,
    ) -> None:
        self.llm_chat      = llm_chat
        self.memory        = memory
        self.logistic_tool = logistic_tool
        self.rag_tool      = rag_tool
        self.router        = QueryRouter(llm_router)

    # ── public entry point ────────────────────────────────────────────────────

    def chat(
        self,
        user_message: str,
        on_step: Optional[Callable[[str, dict], None]] = None,
    ) -> AgentResponse:
        """
        Process a single user message through the full agent pipeline.

        Args:
            user_message: The raw input string from the user.
            on_step     : Optional callback fired at each pipeline stage.
                          Signature: on_step(step_name: str, payload: dict)
                          Step names: 'memory', 'route', 'draft',
                                      'reflecting', 'reflect_result', 'revised'

        Returns:
            An AgentResponse with the final answer and routing metadata.
        """
        def _fire(name: str, payload: dict) -> None:
            """Safely invoke on_step without breaking the pipeline."""
            if on_step:
                try:
                    on_step(name, payload)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("[Orchestrator] on_step callback error: {}", exc)

        t0 = time.time()

        # ── Step 1: Recall memory (ST history + scoped LT profiles only) ───────
        mem_ctx = self.memory.recaller(user_message)
        memory_context = self._format_memory_context(mem_ctx)
        active_profiles = mem_ctx.get("active_profiles") or []
        _fire("memory", {
            "found":    len(active_profiles) > 0,
            "count":    len(active_profiles),
            "profiles": active_profiles,
        })

        # ── Step 2: Route the query ───────────────────────────────────────────
        decision = self.router.route(user_message, memory_context)
        _fire("route", {
            "route":      decision.route,
            "confidence": decision.confidence,
            "reason":     decision.reasoning,
        })

        # ── Step 3: Dispatch to the right tool ───────────────────────────────
        tool_output = self._dispatch(decision, memory_context)

        # ── Step 4: Synthesise initial draft ─────────────────────────────────
        _fire("draft", {"route": decision.route})
        draft = self._synthesise(
            user_message=user_message,
            memory_context=memory_context,
            route=decision.route,
            tool_output=tool_output,
        )

        # ── Step 5 & 6: Reflect → Revise ─────────────────────────────────────
        violated = False
        final_answer = draft

        if active_profiles:
            _fire("reflecting", {"profile_count": len(active_profiles)})
            violation, reason = self._reflect(draft=draft, profiles=active_profiles)
            _fire("reflect_result", {"violated": violation, "reason": reason})

            if violation:
                logger.warning("[Orchestrator] Violation found: '{}'. Revising draft.", reason)
                final_answer = self._revise(draft=draft, reason=reason, profiles=active_profiles)
                violated = True
                _fire("revised", {"reason": reason})
                logger.success("[Orchestrator] Draft revised after reflection.")
            else:
                logger.info("[Orchestrator] Reflection: draft is safe — no revision needed.")
        else:
            logger.debug("[Orchestrator] No active recipient profiles for this turn — skipping reflection.")

        # ── Step 7: Save turn to memory ───────────────────────────────────────
        self.memory.saving_memory(question=user_message, answer=final_answer)

        latency_ms = int((time.time() - t0) * 1000)
        logger.info(
            "[Orchestrator] Done in {}ms | route='{}' | violated={}",
            latency_ms, decision.route, violated,
        )

        return AgentResponse(
            answer=final_answer,
            route=decision.route,
            tool_output=tool_output,
            violated=violated,
            latency_ms=latency_ms,
        )

    # ── internal steps ────────────────────────────────────────────────────────

    def _format_memory_context(self, ctx: dict) -> str:
        """
        Build the MEMORY CONTEXT string for the router + synthesiser.
        Uses only active_profiles (recipient-scoped), not every stored profile.
        """
        try:
            lines = []
            profiles = ctx.get("active_profiles") or []
            if profiles:
                lines.append(
                    "RECIPIENT PROFILES (only for people the user is clearly shopping for "
                    "in this exchange — if empty elsewhere, the user may be buying for themselves):"
                )
                for entry in profiles:
                    lines.append(f"  {entry}")

            if ctx.get("chat_history"):
                lines.append("\nRECENT CONVERSATION:")
                for turn in ctx["chat_history"]:
                    lines.append(f"  {turn['role']}: {turn['content']}")

            return "\n".join(lines)
        except Exception as exc:
            logger.warning("[Orchestrator] Memory context format failed: {}", exc)
            return ""

    def _dispatch(self, decision: RouteDecision, memory_context: str = "") -> str:
        """
        Dispatch the query to the appropriate specialist tool.

        Returns the tool's raw output string, or "" for the direct route.
        """
        route  = decision.route
        params = decision.params or {}

        if route == "logistic":
            if not self.logistic_tool:
                logger.warning("[Orchestrator] Logistic tool not attached.")
                return "Delivery information is currently unavailable."
            question = params.get("district") or params.get("query", "")
            logger.info("[Orchestrator] → LogisticAlert: '{}'", question)
            return self.logistic_tool.generate_answer(question)

        if route == "rag":
            if not self.rag_tool:
                logger.warning("[Orchestrator] RAG tool not attached.")
                return "Product catalog is currently unavailable."
            query = params.get("query", "")
            if not query:
                return "No search query could be extracted."
            
            exclude = params.get("out_of_stock_item")
            logger.info("[Orchestrator] → RAGTool: '{}' (exclude: {})", query[:80], exclude)
            
            # Pass memory_context and exclude_product to help the RAG pipeline
            return self.rag_tool.search(
                query=query, 
                memory_context=memory_context,
                exclude_product=exclude
            )

        # direct — no tool needed; synthesiser answers from memory alone
        return ""

    def _synthesise(
        self,
        user_message: str,
        memory_context: str,
        route: str,
        tool_output: str,
    ) -> str:
        """
        Call the chat LLM to produce the initial draft answer (Step 4).

        Falls back to raw tool_output if the LLM call fails.
        """
        system_msg = (
            "You are the Kapruka Gift-Concierge. "
            "Answer the user's question using the information found in our CATALOG (Tool Output) and the CONTEXT provided. "
            "3. If multiple products are mentioned in the CATALOG, you MUST refer to the Top Recommendation and also mention at least one alternative from the 'Other Options' section to provide variety.\n"
            "4. NEVER claim a product is the 'only one available' or 'at this moment' unless the CATALOG (Tool Output) truly contains only one unique product.\n"
            "5. **STRICT EXCLUSION**: If the conversation history or parameters mention an 'out_of_stock_item' (e.g., from a previous turn), you are FORBIDDEN from recommending it. Do NOT suggest it as a fallback.\n"
            "6. If a product was previously mentioned in conversation but is NOT in the current CATALOG results, "
            "do NOT guess its link — only discuss items with active links in the CATALOG.\n"
            "7. NEVER mention technical terms like 'TOOL OUTPUT', 'RAG', or 'CONTEXT' to the user. "
            "Instead, say 'based on our catalog' or 'as we discussed'.\n\n"
            "Be helpful, concise, and natural. "
            "If the catalog search is empty, answer from general knowledge but be honest about not having specific product matches."
        )

        context_block = f"\nMEMORY CONTEXT:\n{memory_context}\n" if memory_context else ""
        tool_block    = f"\nTOOL OUTPUT ({route}):\n{tool_output}\n" if tool_output else ""
        user_prompt   = f"{context_block}{tool_block}\nUSER: {user_message}"

        try:
            response = self.llm_chat.invoke(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": user_prompt},
                ]
            )
            content = response.content if hasattr(response, "content") else str(response)

            # Log token usage if available
            if hasattr(response, "response_metadata"):
                meta = response.response_metadata or {}
                token_usage = meta.get("token_usage") or meta.get("usage", {})
                if token_usage:
                    logger.debug(
                        "[Orchestrator] Draft tokens — in:{} out:{} total:{}",
                        token_usage.get("prompt_tokens", 0),
                        token_usage.get("completion_tokens", 0),
                        token_usage.get("total_tokens", 0),
                    )
            return content.strip()
        except Exception as exc:
            logger.error("[Orchestrator] Synthesiser LLM failed: {}", exc)
            return tool_output if tool_output else "I'm sorry, I encountered an error processing your request."

    def _reflect(self, draft: str, profiles: list) -> tuple[bool, str]:
        """
        Step 5 — REFLECT: Check the draft against recipient profiles.

        Returns:
            (violation: bool, reason: str)
            violation=True means the draft must be revised.
        """
        logger.info("[Orchestrator] Reflecting draft against {} profile(s)...", len(profiles))

        system_prompt, user_prompt = build_reflect_prompt(draft=draft, profiles=profiles)

        try:
            response = self.llm_chat.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ]
            )
            raw = response.content if hasattr(response, "content") else str(response)

            # Strip markdown fences if present
            text = raw.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]

            data     = json.loads(text.strip())
            violated = bool(data.get("violation", False))
            reason   = data.get("reason", "")
            logger.info("[Orchestrator] Reflect result — violation={} reason='{}'", violated, reason)
            return violated, reason

        except Exception as exc:
            # If reflection fails, be safe and pass (no revision)
            logger.warning("[Orchestrator] Reflect step failed (skipping): {}", exc)
            return False, ""

    def _revise(self, draft: str, reason: str, profiles: list) -> str:
        """
        Step 6 — REVISE: Rewrite the draft to fix the detected violation.

        Returns the revised answer string.
        """
        logger.info("[Orchestrator] Revising draft to fix: '{}'", reason)

        system_prompt, user_prompt = build_revise_prompt(
            draft=draft, reason=reason, profiles=profiles
        )

        try:
            response = self.llm_chat.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ]
            )
            content = response.content if hasattr(response, "content") else str(response)
            return content.strip()
        except Exception as exc:
            logger.error("[Orchestrator] Revise step failed: {}", exc)
            return draft  # fall back to original draft

    # ── helpers ───────────────────────────────────────────────────────────────

    def _model_name(self) -> str:
        """Return the chat LLM model name for logging."""
        if hasattr(self.llm_chat, "model_name"):
            return self.llm_chat.model_name
        if hasattr(self.llm_chat, "model"):
            return self.llm_chat.model
        return "unknown"
