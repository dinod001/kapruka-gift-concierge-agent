"""
Long-Term Memory Store
======================
Saves and loads recipient preference profiles in the format:

    [
        {"wife": {"allergy": "nuts"}},
        {"mother": {"likes": "fruit"}}
    ]

Each entry is a dict whose single key is the recipient name and whose
value is a free-form dict of traits (allergy, likes, dislikes, etc.)
"""

import json
from pathlib import Path
from typing import Any

from loguru import logger

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate

from src.memory.memory_template import EXTRACTION_PROMPT


class LongTermMemory:
    """
    Manages recipient preference profiles persisted as a JSON file.

    Schema
    ------
    The JSON file is a list of single-key dicts:
        [
            {"<recipient_name>": {<trait_key>: <trait_value>, ...}},
            ...
        ]
    """

    def __init__(self, data_path: str, llm=None):
        """
        Parameters
        ----------
        data_path : str
            Absolute or relative path to the JSON profile file.
        llm : LangChain LLM instance, optional
            Required only when calling `extract_and_save`.
        """
        self.data_path = Path(data_path)
        self.llm = llm

    # ------------------------------------------------------------------
    # READ
    # ------------------------------------------------------------------

    def get_profiles(self) -> list[dict[str, Any]]:
        """
        Load and return all recipient profiles from disk.

        Returns
        -------
        list[dict]
            E.g. [{"wife": {"allergy": "nuts"}}, ...]
            Returns [] if the file is missing or empty.
        """
        try:
            if not self.data_path.exists() or self.data_path.stat().st_size == 0:
                logger.debug("[LongTermMemory] Profile file empty or missing — returning []")
                return []
            with open(self.data_path, "r", encoding="utf-8") as f:
                profiles = json.load(f)
            logger.debug(f"[LongTermMemory] Loaded {len(profiles)} profile(s) from {self.data_path}")
            return profiles
        except FileNotFoundError:
            logger.warning(f"[LongTermMemory] File not found: {self.data_path}")
            return []
        except json.JSONDecodeError:
            logger.error(f"[LongTermMemory] Invalid JSON in: {self.data_path}")
            return []

    def get_profile(self, recipient: str) -> dict[str, Any] | None:
        """
        Fetch the profile dict for a single named recipient.

        Parameters
        ----------
        recipient : str
            E.g. "wife"

        Returns
        -------
        dict | None
            E.g. {"allergy": "nuts"}, or None if not found.
        """
        for entry in self.get_profiles():
            if recipient in entry:
                return entry[recipient]
        return None

    # ------------------------------------------------------------------
    # WRITE
    # ------------------------------------------------------------------

    def save_profiles(self, profiles: list[dict[str, Any]]) -> None:
        """
        Persist a full profile list to disk (pretty-printed JSON).

        Parameters
        ----------
        profiles : list[dict]
            E.g. [{"wife": {"allergy": "nuts"}}, {"mother": {"likes": "fruit"}}]
        """
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.data_path, "w", encoding="utf-8") as f:
            json.dump(profiles, f, indent=4, ensure_ascii=False)
        logger.success(f"[LongTermMemory] Saved {len(profiles)} profile(s) → {self.data_path}")

    def upsert_profile(self, recipient: str, traits: dict[str, Any]) -> None:
        """
        Add or update a single recipient's traits and save to disk.

        Parameters
        ----------
        recipient : str   – e.g. "wife"
        traits    : dict  – e.g. {"allergy": "nuts", "likes": "chocolates"}
        """
        profiles = self.get_profiles()
        for entry in profiles:
            if recipient in entry:
                entry[recipient].update(traits)
                self.save_profiles(profiles)
                return
        # New recipient not yet in the list
        profiles.append({recipient: traits})
        self.save_profiles(profiles)

    # ------------------------------------------------------------------
    # LLM-POWERED EXTRACTION
    # ------------------------------------------------------------------

    def extract_and_save(self, conversation_text: str) -> list[dict[str, Any]]:
        """
        Use the LLM to extract recipient profiles from raw conversation
        text and persist them.

        Parameters
        ----------
        conversation_text : str
            Raw user message or conversation snippet.

        Returns
        -------
        list[dict]
            The extracted (and saved) profile list.
        """
        if self.llm is None:
            logger.error("[LongTermMemory] No LLM provided — cannot run extraction.")
            raise ValueError("[LongTermMemory] No LLM provided for extraction.")
        logger.info("[LongTermMemory] Running LLM profile extraction...")

        prompt = PromptTemplate(
            input_variables=["conversation"],
            template=EXTRACTION_PROMPT,
        )
        chain = prompt | self.llm | JsonOutputParser()
        profiles: list[dict[str, Any]] = chain.invoke({"conversation": conversation_text})

        self.save_profiles(profiles)
        return profiles
