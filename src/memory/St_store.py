"""
Short-Term Memory Store
=======================
Manages the in-session conversation history (chat turns) in memory.

Each turn is stored as a dict:
    {"role": "user" | "assistant", "content": "<message>"}

The buffer is kept in-memory during a session and can be cleared
or trimmed to a fixed window size.
"""

from typing import Any

from loguru import logger


class ShortTermMemory:
    """
    In-memory rolling buffer of conversation turns for the current session.

    Schema
    ------
    Internal buffer is a list of dicts:
        [
            {"role": "user",      "content": "I need a gift for my wife."},
            {"role": "assistant", "content": "Sure! Does she have any allergies?"},
            ...
        ]
    """

    def __init__(self, max_turns: int = 20):
        """
        Parameters
        ----------
        max_turns : int
            Maximum number of turns (user + assistant messages combined)
            to keep in the buffer. Oldest turns are dropped when exceeded.
            Default: 20.
        """
        self.max_turns = max_turns
        self._buffer: list[dict[str, str]] = []

    # ------------------------------------------------------------------
    # READ
    # ------------------------------------------------------------------

    def get_history(self) -> list[dict[str, str]]:
        """
        Return all turns currently in the buffer.

        Returns
        -------
        list[dict]
            E.g. [{"role": "user", "content": "..."}, ...]
        """
        return list(self._buffer)

    def get_context_string(self) -> str:
        """
        Return the history as a plain multi-line string suitable for
        injecting into a prompt.

        Returns
        -------
        str
            E.g. "user: ...\nassistant: ...\n"
        """
        return "\n".join(
            f"{turn['role']}: {turn['content']}" for turn in self._buffer
        )

    def last_user_message(self) -> str | None:
        """Return the most recent user message, or None if buffer is empty."""
        for turn in reversed(self._buffer):
            if turn["role"] == "user":
                return turn["content"]
        return None

    # ------------------------------------------------------------------
    # WRITE
    # ------------------------------------------------------------------

    def add_turn(self, role: str, content: str) -> None:
        """
        Append a single turn to the buffer.

        Parameters
        ----------
        role    : str – "user" or "assistant"
        content : str – the message text
        """
        if role not in ("user", "assistant"):
            logger.error(f"[ShortTermMemory] Invalid role '{role}'. Must be 'user' or 'assistant'.")
            raise ValueError(f"[ShortTermMemory] Invalid role '{role}'. Use 'user' or 'assistant'.")
        self._buffer.append({"role": role, "content": content})
        logger.debug(f"[ShortTermMemory] Added '{role}' turn. Buffer size: {len(self._buffer)}")
        self._trim()

    def add_user(self, content: str) -> None:
        """Convenience: append a user turn."""
        self.add_turn("user", content)

    def add_assistant(self, content: str) -> None:
        """Convenience: append an assistant turn."""
        self.add_turn("assistant", content)

    def clear(self) -> None:
        """Wipe the entire buffer (start a fresh session)."""
        self._buffer = []
        logger.info("[ShortTermMemory] Buffer cleared — new session started.")

    # ------------------------------------------------------------------
    # INTERNAL
    # ------------------------------------------------------------------

    def _trim(self) -> None:
        """Drop oldest turns if buffer exceeds max_turns."""
        if len(self._buffer) > self.max_turns:
            dropped = len(self._buffer) - self.max_turns
            self._buffer = list(self._buffer[-self.max_turns:])
            logger.debug(f"[ShortTermMemory] Trimmed {dropped} oldest turn(s). Max={self.max_turns}")

