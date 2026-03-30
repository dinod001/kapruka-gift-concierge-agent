"""
Recipient scope for long-term profiles.

Only inject stored recipient profiles when the user (or recent user turns)
clearly refers to that person. If the user says the gift is "for me" / "for myself",
do not load other recipients' traits — that prevents e.g. a "sister" profile from
colouring a self-purchase or a vague follow-up like "no vanilla".
"""

from __future__ import annotations

import re
from typing import Any


def _profile_keys(all_profiles: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    for entry in all_profiles:
        keys.extend(list(entry.keys()))
    return keys


def _filter_entries(
    all_profiles: list[dict[str, Any]],
    mentioned: set[str],
) -> list[dict[str, Any]]:
    if not mentioned:
        return []
    out: list[dict[str, Any]] = []
    for entry in all_profiles:
        for k in entry.keys():
            if k in mentioned:
                out.append(entry)
                break
    return out


def _mentions_named_recipients(text: str, keys: list[str]) -> set[str]:
    """Return profile keys (original casing) explicitly referenced in text."""
    mentioned: set[str] = set()
    t = text.lower()
    for key in keys:
        k = key.lower().strip()
        if not k:
            continue
        patterns = [
            rf"\bmy\s+{re.escape(k)}\b",
            rf"\bfor\s+(?:my\s+)?{re.escape(k)}\b",
            rf"\b{re.escape(k)}'s\b",
            rf"\b{re.escape(k)}\b",  # last: short names like "sister"
        ]
        if any(re.search(p, t) for p in patterns):
            mentioned.add(key)
    return mentioned


_SELF_SCOPE_RE = re.compile(
    r"(?i)"
    r"\bfor\s+myself\b|"
    r"\bbuy(?:ing)?\s+(?:[a-z]+\s+){0,6}for\s+me\b|"
    r"\bget(?:ting)?\s+(?:[a-z]+\s+){0,6}for\s+me\b|"
    r"\bgift\s+for\s+myself\b|"
    r"\bfor\s+me\b|"
    r"\bi\s+want\b[\s\S]{0,140}?\bfor\s+me\b|"
    r"\bi'?d\s+like\b[\s\S]{0,140}?\bfor\s+me\b"
)


def _text_establishes_self_scope(text: str) -> bool:
    return bool(_SELF_SCOPE_RE.search(text))


def resolve_active_profiles(
    question: str,
    chat_history: list[dict[str, str]],
    all_profiles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Decide which stored profiles may influence this turn.

    Search order: current user message, then older user messages (newest first).
    The first scope decision wins:
    - If a message establishes self-scope ("for me", etc.), return only profiles
      explicitly named in that same message alongside self (usually none).
    - If a message names a recipient, return only those entries.
    - If nothing matches, return [] (do not dump every profile).
    """
    keys = _profile_keys(all_profiles)
    if not keys:
        return []

    user_turns = [t["content"] for t in chat_history if t.get("role") == "user"]
    search_texts = [question] + list(reversed(user_turns))

    for text in search_texts:
        if not (text and text.strip()):
            continue
        if _text_establishes_self_scope(text):
            mentioned = _mentions_named_recipients(text, keys)
            return _filter_entries(all_profiles, mentioned)
        mentioned = _mentions_named_recipients(text, keys)
        if mentioned:
            return _filter_entries(all_profiles, mentioned)

    return []
