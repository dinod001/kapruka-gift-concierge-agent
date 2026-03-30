"""
Memory Layer Test Script
========================
Tests:
  1. ShortTermMemory  — add turns, trim, clear
  2. LongTermMemory   — manual save / load / upsert + LLM extraction
  3. MemoryDistiller  — recaller, saving_memory, distillation trigger (intent-only)

Run from project root:
    python -m tests.test_memory
"""

import sys
import os

# ── project root on sys.path ─────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT     = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_ROOT)

# ── load .env before any config / LLM imports ────────────────────────────────
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from loguru import logger
from memory.St_store import ShortTermMemory
from memory.lt_store import LongTermMemory
from memory.memory_ops import MemoryDistiller

# ── test profile file (safe to delete after) ─────────────────────────────────
TEST_PROFILE_PATH = os.path.join(PROJECT_ROOT, "data", "test_recipient_profiles.json")

PASS = "✅ PASS"
FAIL = "❌ FAIL"


# =============================================================================
# 1. SHORT-TERM MEMORY
# =============================================================================

def test_short_term_memory():
    logger.info("=" * 60)
    logger.info("TEST 1 — ShortTermMemory")
    logger.info("=" * 60)

    stm = ShortTermMemory(max_turns=4)

    # 1a. Add turns
    stm.add_user("I need a gift for my wife.")
    stm.add_assistant("Sure! Any allergies or preferences?")
    stm.add_user("She's allergic to nuts.")
    stm.add_assistant("Got it — no nuts. Let me find something.")

    history = stm.get_history()
    assert len(history) == 4, f"Expected 4 turns, got {len(history)}"
    logger.success(f"  {PASS}  add turns: {len(history)} turns stored")

    # 1b. Trimming — 5th turn drops the oldest
    stm.add_user("Also she likes chocolates.")
    history = stm.get_history()
    assert len(history) == 4, f"Expected 4 after trim, got {len(history)}"
    logger.success(f"  {PASS}  trim: buffer stayed at max_turns=4")

    # 1c. Context string
    ctx = stm.get_context_string()
    assert "assistant" in ctx and "user" in ctx
    logger.success(f"  {PASS}  get_context_string: contains role labels")

    # 1d. Last user message
    last = stm.last_user_message()
    assert last == "Also she likes chocolates.", f"Unexpected last user msg: {last}"
    logger.success(f"  {PASS}  last_user_message: '{last}'")

    # 1e. Clear
    stm.clear()
    assert stm.get_history() == []
    logger.success(f"  {PASS}  clear: buffer is empty")


# =============================================================================
# 2. LONG-TERM MEMORY  (manual, no LLM)
# =============================================================================

def test_long_term_memory_manual():
    logger.info("=" * 60)
    logger.info("TEST 2a — LongTermMemory (manual save / load / upsert)")
    logger.info("=" * 60)

    ltm = LongTermMemory(data_path=TEST_PROFILE_PATH)

    # 2a. Save
    profiles = [
        {"wife":   {"allergy": "nuts"}},
        {"mother": {"likes": "fruit"}},
    ]
    ltm.save_profiles(profiles)
    logger.success(f"  {PASS}  save_profiles: wrote {len(profiles)} profiles")

    # 2b. Load
    loaded = ltm.get_profiles()
    assert loaded == profiles, f"Mismatch: {loaded}"
    logger.success(f"  {PASS}  get_profiles: loaded correctly")

    # 2c. Get single profile
    wife = ltm.get_profile("wife")
    assert wife == {"allergy": "nuts"}, f"Unexpected: {wife}"
    logger.success(f"  {PASS}  get_profile('wife'): {wife}")

    # 2d. Upsert — update existing
    ltm.upsert_profile("wife", {"likes": "flowers"})
    wife = ltm.get_profile("wife")
    assert wife == {"allergy": "nuts", "likes": "flowers"}, f"Unexpected: {wife}"
    logger.success(f"  {PASS}  upsert existing: {wife}")

    # 2e. Upsert — add new recipient
    ltm.upsert_profile("friend", {"likes": "books"})
    friend = ltm.get_profile("friend")
    assert friend == {"likes": "books"}, f"Unexpected: {friend}"
    logger.success(f"  {PASS}  upsert new recipient: {friend}")

    # 2f. Missing recipient → None
    missing = ltm.get_profile("nobody")
    assert missing is None
    logger.success(f"  {PASS}  get_profile('nobody'): returned None")


# =============================================================================
# 3. LONG-TERM MEMORY  (LLM extraction)
# =============================================================================

def test_long_term_memory_llm():
    logger.info("=" * 60)
    logger.info("TEST 2b — LongTermMemory (LLM extraction via Groq)")
    logger.info("=" * 60)

    from infrastructure.llm.llm_provider import get_extractor_llm
    llm = get_extractor_llm(temperature=0)
    ltm = LongTermMemory(data_path=TEST_PROFILE_PATH, llm=llm)

    conversation = (
        "I need a gift for my wife — she is allergic to nuts and loves chocolates. "
        "My mother is visiting next week and she really enjoys fresh fruit."
    )

    profiles = ltm.extract_and_save(conversation)
    logger.info(f"  Extracted profiles: {profiles}")

    assert isinstance(profiles, list), "Expected a list"
    assert len(profiles) > 0, "Expected at least one profile extracted"

    keys = [list(p.keys())[0] for p in profiles]
    logger.info(f"  Recipients found: {keys}")

    assert any(k in ("wife", "mother") for k in keys), \
        f"Expected 'wife' or 'mother' in extracted keys, got: {keys}"

    logger.success(f"  {PASS}  LLM extraction: found recipients {keys}")


# =============================================================================
# 4. MEMORY DISTILLER
# =============================================================================

def test_memory_distiller():
    logger.info("=" * 60)
    logger.info("TEST 3 — MemoryDistiller")
    logger.info("=" * 60)

    from infrastructure.llm.llm_provider import get_extractor_llm
    llm = get_extractor_llm(temperature=0)
    
    # Initialize with LLM and test profile path
    mem = MemoryDistiller(llm=llm, profile_path=TEST_PROFILE_PATH)

    # 4a. Intent trigger — even with empty buffer, keyword should work
    assert mem.should_distill("Please remember my wife hates nuts") is True
    logger.success(f"  {PASS}  should_distill (intent keyword): True")

    # 4b. Saving memory — intent triggered case
    mem.saving_memory("Remember that my sister likes coffee", "Got it, I'll remember your sister likes coffee.")
    # Recaller should see the turn in history
    ctx = mem.recaller("What does my sister like?")
    assert len(ctx["chat_history"]) == 2
    logger.success(f"  {PASS}  saving_memory: turns stored correctly")

    # 4c. saving_memory — no trigger case
    mem.saving_memory("How are you today?", "I'm doing great, thank you!")
    assert len(mem.get_chat_history()) == 4
    logger.success(f"  {PASS}  saving_memory: total 4 turns in buffer")

    # 4d. recaller returns correct keys
    assert "chat_history" in ctx
    assert "all_profiles" in ctx
    assert "active_profiles" in ctx
    assert "question" in ctx
    logger.success(f"  {PASS}  recaller: returned context keys")

    # 4e. clear_session
    mem.clear_session()
    assert mem.get_chat_history() == []
    logger.success(f"  {PASS}  clear_session: buffer empty")


# =============================================================================
# RUNNER
# =============================================================================

def run_all():
    results = {}

    for name, fn in [
        ("ShortTermMemory",             test_short_term_memory),
        ("LongTermMemory (manual)",     test_long_term_memory_manual),
        ("LongTermMemory (LLM/Groq)",   test_long_term_memory_llm),
        ("MemoryDistiller",             test_memory_distiller),
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
        logger.info(f"  {result}  {name}")

    # cleanup test file
    import pathlib
    p = pathlib.Path(TEST_PROFILE_PATH)
    if p.exists():
        p.unlink()
        logger.debug(f"  Cleaned up test file: {p}")


if __name__ == "__main__":
    run_all()
