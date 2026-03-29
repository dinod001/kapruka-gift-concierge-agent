"""
Logistics Agent Test Script
============================
Tests the LogisticAlert agent for delivery feasibility across
Sri Lankan districts using the Groq extractor LLM.

Run from project root:
    python -m tests.test_logistic_agent
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
from infrastructure.llm.llm_provider import get_extractor_llm
from agents.tools.logsitic_agent import LogisticAlert

PASS = "✅ PASS"
FAIL = "❌ FAIL"

# ── Test questions covering all zone types ────────────────────────────────────
TEST_CASES = [
    # (district_question,                        expected_keyword_in_response)
    ("Can you deliver to Colombo?",               "express"),
    ("Is delivery available for Kandy district?", "standard"),
    ("Can Kapruka deliver to Jaffna?",            "extended"),
    ("What about delivery to Monaragala?",        "extended"),
    ("Is delivery possible to Nuwaraeliya?",      "standard"),
    ("Can you deliver to Mars?",                  "outside"),   # should say not feasible
]


def run_tests():
    llm   = get_extractor_llm(temperature=0)
    agent = LogisticAlert(llm=llm)

    results = {}

    logger.info("=" * 60)
    logger.info("Logistics Agent — Delivery Feasibility Tests")
    logger.info("=" * 60)

    for question, expected_keyword in TEST_CASES:
        logger.info(f"\n  Q: {question}")
        try:
            answer = agent.generate_answer(question)
            logger.info(f"  A: {answer.strip()}")

            if expected_keyword.lower() in answer.lower():
                results[question] = PASS
                logger.success(f"  {PASS}  (keyword '{expected_keyword}' found)")
            else:
                results[question] = f"⚠️  WARN — expected '{expected_keyword}' not found (manual review needed)"
                logger.warning(f"  ⚠️  expected keyword '{expected_keyword}' not in response")

        except Exception as e:
            results[question] = f"❌ FAIL — {e}"
            logger.error(f"  ❌ FAIL: {e}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    for q, r in results.items():
        logger.info(f"  {r}  |  {q}")


if __name__ == "__main__":
    run_tests()
