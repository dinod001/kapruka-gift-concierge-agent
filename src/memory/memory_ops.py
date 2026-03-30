from memory.lt_store import LongTermMemory
from memory.recipient_scope import resolve_active_profiles
from memory.St_store import ShortTermMemory
from loguru import logger
from typing import Any


class MemoryDistiller:
    """
    Coordinates short-term and long-term memory operations.
    Can be used to check if new information should be distilled into long-term memory.
    """

    def __init__(self, llm=None, profile_path: str = "data/recipient_profiles.json") -> None:
        """
        Args:
            llm: The LLM provider (e.g., Groq) used for extraction.
            profile_path: Path to the JSON file where profiles are stored.
        """
        self.long_term  = LongTermMemory(data_path=profile_path, llm=llm)
        self.short_term = ShortTermMemory(max_turns=20)
        logger.debug(f"[MemoryDistiller] Initialized with profile path: {profile_path}")

    # ------------------------------------------------------------------
    # DISTILLATION TRIGGER
    # ------------------------------------------------------------------

    def should_distill(self, question: str) -> bool:
        """
        Check if distillation should be triggered based STRICTLY on intent.
        
        Distillation is triggered if the question contains memory-intent keywords.
        """
        keywords = ["remember", "from now on", "remind me", "always", "never", "allergic", "likes", "dislikes", "loves", "hates"]
        content_lower = question.lower()
        
        if any(kw in content_lower for kw in keywords):
            logger.info(f"[MemoryDistiller] Intent detected in: '{question}'")
            return True

        return False

    # ------------------------------------------------------------------
    # RECALL  — fetch context from both LT and ST memory
    # ------------------------------------------------------------------

    def recaller(self, question: str) -> dict[str, Any]:
        """
        Retrieve the full memory context needed to answer a question.
        Returns a dictionary for prompt injection.
        """
        logger.info(f"[MemoryDistiller] Recalling context for: '{question}'")

        chat_history = self.short_term.get_history()
        all_profiles = self.long_term.get_profiles()
        active = resolve_active_profiles(question, chat_history, all_profiles)

        return {
            "chat_history":    chat_history,
            "all_profiles":    all_profiles,
            "active_profiles": active,
            "question":        question,
        }

    # ------------------------------------------------------------------
    # SHORT-TERM HELPERS
    # ------------------------------------------------------------------

    def get_chat_history(self) -> list[dict[str, str]]:
        """Return all turns currently in the short-term buffer."""
        return self.short_term.get_history()

    def load_chat_history(self, history: list[dict[str, str]]) -> None:
        """Overwrite the short-term buffer with provided history."""
        self.short_term.set_history(history)

    def clear_session(self) -> None:
        """Wipe the short-term buffer (call at session end)."""
        self.short_term.clear()

    # ------------------------------------------------------------------
    # SAVE  — persist latest conversation to ST and LT if needed
    # ------------------------------------------------------------------

    def saving_memory(self, question: str, answer: str) -> None:
        """
        Save the latest user/assistant turns to short-term memory.
        If distillation is triggered by keywords, also extract & persist profiles.
        """
        # Always update short-term buffer
        self.short_term.add_user(question)
        self.short_term.add_assistant(answer)
        logger.info("[MemoryDistiller] Short-term buffer updated (user + assistant turns).")

        # Intent-based distillation
        if self.should_distill(question):
            logger.info("[MemoryDistiller] Distillation triggered — extracting profiles.")
            try:
                # Extract and persist profiles from the current question only
                # as requested by the user.
                profiles = self.long_term.extract_and_save(question)
                logger.success(f"[MemoryDistiller] Distilled {len(profiles)} profile(s) from latest question.")

            except Exception as e:
                logger.warning(f"[MemoryDistiller] Extraction failed or skipped: {e}")
