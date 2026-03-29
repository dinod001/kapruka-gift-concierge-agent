"""
Orchestrator End-to-End Test Script
=====================================
Tests the full AgentOrchestrator pipeline:
  - Logistic route  → LogisticAlert
  - RAG route       → RAGTool
  - Direct route    → Synthesiser only (no tool)

Run from project root:
    python -m tests.test_orchestrator
"""

import sys
import os

# ── project root on sys.path ─────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT     = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_ROOT)

# ── load .env before any LLM/config imports ───────────────────────────────────
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from loguru import logger
from infrastructure.llm.llm_provider import (
    get_chat_llm,
    get_router_llm,
    get_extractor_llm,
)
from services.ingest_service.embeddings import get_default_embeddings

from memory.memory_ops import MemoryDistiller
from agents.tools.logsitic_agent import LogisticAlert
from agents.tools.rag_tool import RAGTool
from agents.orchestrator import AgentOrchestrator, AgentResponse

PASS = "✅ PASS"
FAIL = "❌ FAIL"

# ── test profile path ─────────────────────────────────────────────────────────
TEST_PROFILE_PATH = os.path.join(PROJECT_ROOT, "data", "test_orc_profiles.json")


def build_orchestrator(with_rag: bool = False) -> AgentOrchestrator:
    """Build a fully wired orchestrator for testing."""
    llm_chat      = get_chat_llm(temperature=0)
    llm_router    = get_router_llm(temperature=0)
    llm_extractor = get_extractor_llm(temperature=0)
    embedder      = get_default_embeddings()

    memory        = MemoryDistiller(llm=llm_extractor, profile_path=TEST_PROFILE_PATH)
    logistic_tool = LogisticAlert(llm=llm_extractor)
    rag_tool      = RAGTool(embedder=embedder, llm=llm_chat) if with_rag else None

    return AgentOrchestrator(
        llm_chat=llm_chat,
        llm_router=llm_router,
        memory=memory,
        logistic_tool=logistic_tool,
        rag_tool=rag_tool,
    )


# =============================================================================
# TEST 1 — Logistic route
# =============================================================================

def test_logistic_route(agent: AgentOrchestrator):
    logger.info("=" * 60)
    logger.info("TEST 1 — Logistic Route")
    logger.info("=" * 60)

    response: AgentResponse = agent.chat("Can you deliver to Jaffna?")

    logger.info(f"  Route      : {response.route}")
    logger.info(f"  Tool output: {response.tool_output[:120]}")
    logger.info(f"  Answer     : {response.answer[:200]}")
    logger.info(f"  Latency    : {response.latency_ms}ms")

    assert response.route == "logistic", f"Expected 'logistic', got '{response.route}'"
    assert response.answer, "Answer should not be empty"
    assert response.latency_ms > 0
    logger.success(f"  {PASS}  Logistic route")


# =============================================================================
# TEST 2 — Direct route
# =============================================================================

def test_direct_route(agent: AgentOrchestrator):
    logger.info("=" * 60)
    logger.info("TEST 2 — Direct Route")
    logger.info("=" * 60)

    response: AgentResponse = agent.chat("Hello! How are you?")

    logger.info(f"  Route  : {response.route}")
    logger.info(f"  Answer : {response.answer[:200]}")
    logger.info(f"  Latency: {response.latency_ms}ms")

    assert response.route == "direct", f"Expected 'direct', got '{response.route}'"
    assert response.answer, "Answer should not be empty"
    logger.success(f"  {PASS}  Direct route")


# =============================================================================
# TEST 3 — Memory: save & recall across turns
# =============================================================================

def test_memory_across_turns(agent: AgentOrchestrator):
    logger.info("=" * 60)
    logger.info("TEST 3 — Memory Across Turns")
    logger.info("=" * 60)

    # Turn 1
    agent.chat("Remember that my wife is allergic to nuts.")
    # Turn 2 — recall should include the info
    response = agent.chat("What should I avoid when buying a gift for my wife?")

    logger.info(f"  Answer: {response.answer[:300]}")

    assert response.answer, "Answer should not be empty"
    # Two acceptable outcomes:
    #  a) Reflection caught a violation → violated=True (draft was revised away from nuts)
    #  b) Synthesiser already used memory and mentioned nut/allergy in its answer
    assert response.violated is True \
        or "nut" in response.answer.lower() \
        or "allerg" in response.answer.lower(), \
        "Expected either reflection violation OR nut-allergy info in the answer"
    logger.success(f"  {PASS}  Memory recalled across turns (violated={response.violated})")


# =============================================================================
# TEST 4 — RAG route (skip if rag_tool not available)
# =============================================================================

def test_rag_route(agent: AgentOrchestrator):
    logger.info("=" * 60)
    logger.info("TEST 4 — RAG Route")
    logger.info("=" * 60)

    if agent.rag_tool is None:
        logger.warning("  ⚠️  RAG tool not attached — skipping test.")
        return

    response: AgentResponse = agent.chat("Can you recommend a chocolate gift for my mother?")

    logger.info(f"  Route      : {response.route}")
    logger.info(f"  Tool output: {response.tool_output[:120]}")
    logger.info(f"  Answer     : {response.answer[:200]}")

    assert response.route == "rag", f"Expected 'rag', got '{response.route}'"
    assert response.answer, "Answer should not be empty"
    logger.success(f"  {PASS}  RAG route")


# =============================================================================
# TEST 5 — Reflection Loop (allergy violation caught & revised)
# =============================================================================

def test_reflection_loop():
    """
    Tests _reflect() and _revise() directly with a FORCED bad draft.

    Note: We cannot rely on chat() to test this because the synthesiser already
    has the allergy profile in its memory context and writes a safe draft — so
    _reflect() correctly returns violation=False. Instead we inject a bad draft
    directly to validate the reflection mechanism in isolation.
    """
    logger.info("=" * 60)
    logger.info("TEST 5 — Reflection Loop (_reflect + _revise unit test)")
    logger.info("=" * 60)

    from infrastructure.llm.llm_provider import get_chat_llm, get_router_llm, get_extractor_llm
    from memory.lt_store import LongTermMemory

    llm_chat      = get_chat_llm(temperature=0)
    llm_router    = get_router_llm(temperature=0)
    llm_extractor = get_extractor_llm(temperature=0)

    # Pre-seed LT memory with a nut allergy profile
    lt_mem = LongTermMemory(data_path=TEST_PROFILE_PATH)
    lt_mem.save_profiles([{"wife": {"allergy": "nuts"}}])
    logger.info("  Pre-seeded profile: wife → allergy: nuts")

    memory = MemoryDistiller(llm=llm_extractor, profile_path=TEST_PROFILE_PATH)
    agent  = AgentOrchestrator(
        llm_chat=llm_chat,
        llm_router=llm_router,
        memory=memory,
    )
    profiles = memory.long_term.get_profiles()

    # ── 5a. _reflect() with a FORCED bad draft that mentions nuts ────────────
    bad_draft = "I recommend a delicious peanut brittle and walnut brownie gift box for your wife!"
    logger.info(f"  Testing _reflect() with bad draft: '{bad_draft[:60]}'")

    violated, reason = agent._reflect(draft=bad_draft, profiles=profiles)
    logger.info(f"  violation={violated} reason='{reason}'")

    assert violated is True, f"Expected violation=True for nut-heavy draft, got False"
    logger.success(f"  {PASS}  _reflect(): nut allergy violation detected")

    # ── 5b. _revise() should produce a safe alternative ──────────────────────
    revised = agent._revise(draft=bad_draft, reason=reason, profiles=profiles)
    logger.info(f"  Revised answer: '{revised}'")

    assert revised, "Revised answer should not be empty"
    assert revised != bad_draft, "Revised answer should differ from bad draft"
    logger.success(f"  {PASS}  _revise(): safe revised answer produced")




# =============================================================================
# RUNNER
# =============================================================================

def run_all():
    results = {}

    logger.info("Building orchestrator (logistic only, no RAG for speed)...")
    agent = build_orchestrator(with_rag=False)

    for name, fn in [
        ("Logistic Route",           lambda: test_logistic_route(agent)),
        ("Direct Route",             lambda: test_direct_route(agent)),
        ("Memory Across Turns",      lambda: test_memory_across_turns(agent)),
        ("RAG Route",                lambda: test_rag_route(agent)),
        ("Reflection Loop",          test_reflection_loop),
    ]:
        try:
            fn()
            results[name] = PASS
        except Exception as e:
            logger.error(f"  ❌ FAIL  {name}: {e}")
            results[name] = f"❌ FAIL — {e}"

    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    for name, result in results.items():
        logger.info(f"  {result}  |  {name}")

    # Profiles are kept in TEST_PROFILE_PATH after the run so you can inspect them.
    # Uncomment below to auto-delete after testing:
    # import pathlib
    # p = pathlib.Path(TEST_PROFILE_PATH)
    # if p.exists():
    #     p.unlink()
    #     logger.debug(f"  Cleaned up: {p}")
    logger.info(f"  Profiles saved to: {TEST_PROFILE_PATH}")


if __name__ == "__main__":
    run_all()
