"""Unit tests for recipient profile scoping (LT memory injection)."""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_ROOT)

from memory.recipient_scope import resolve_active_profiles  # noqa: E402

_STORE = [{"sister": {"likes": "cakes"}}, {"wife": {"allergy": "nuts"}}]


def test_self_purchase_excludes_profiles():
    hist = []
    assert resolve_active_profiles("I want a cake for me", hist, _STORE) == []


def test_self_then_follow_up_still_empty():
    hist = [
        {"role": "user", "content": "I want to buy a cake for me"},
        {"role": "assistant", "content": "Here is a chocolate slice."},
    ]
    assert resolve_active_profiles("No vanilla one please", hist, _STORE) == []


def test_explicit_sister_gets_profile():
    hist = []
    active = resolve_active_profiles("Gift for my sister", hist, _STORE)
    assert len(active) == 1 and "sister" in active[0]


def test_sticky_recipient_from_history():
    hist = [
        {"role": "user", "content": "Something sweet for my sister"},
        {"role": "assistant", "content": "How about cake?"},
    ]
    active = resolve_active_profiles("Without cinnamon", hist, _STORE)
    assert len(active) == 1 and "sister" in active[0]


def test_self_overrides_in_latest_message():
    hist = [
        {"role": "user", "content": "Gift for my sister"},
        {"role": "assistant", "content": "Sure."},
    ]
    assert resolve_active_profiles("Actually this one is for me", hist, _STORE) == []


def test_vague_question_no_profile_dump():
    hist = []
    assert resolve_active_profiles("Do you sell flowers?", hist, _STORE) == []


if __name__ == "__main__":
    test_self_purchase_excludes_profiles()
    test_self_then_follow_up_still_empty()
    test_explicit_sister_gets_profile()
    test_sticky_recipient_from_history()
    test_self_overrides_in_latest_message()
    test_vague_question_no_profile_dump()
    print("recipient_scope: OK")
