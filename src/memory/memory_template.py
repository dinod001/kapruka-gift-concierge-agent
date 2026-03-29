"""
Memory Prompt Templates
=======================
Prompts used by the memory layer to extract / update recipient profiles
from free-form conversation text.
"""

# ============================================================
# Recipient-Profile Extraction Prompt
# ============================================================

EXTRACTION_PROMPT = """You are a memory extraction assistant for a gift-concierge service.

Your task is to read the conversation snippet below and extract any information
about gift recipients (e.g. family members, friends) – their preferences,
allergies, dislikes, or any other relevant traits.

OUTPUT FORMAT (strict JSON, no markdown fences):
Return a JSON array where every element is an object with ONE key:
  - The key   → recipient name  (e.g. "wife", "mother", "friend")
  - The value → a flat object containing the extracted traits

Allowed trait keys (use only what is found in the text):
  - "allergy"   : foods / substances the recipient must avoid
  - "likes"     : things the recipient enjoys
  - "dislikes"  : things the recipient does not enjoy
  - "occasion"  : special occasion (birthday, anniversary, etc.)
  - "age"       : approximate age
  - "note"      : any other relevant detail

Example valid output:
[
    {{"wife": {{"allergy": "nuts", "likes": "chocolates"}}}},
    {{"mother": {{"likes": "fruit", "dislikes": "spicy food"}}}}
]

Rules:
1. Only include recipients explicitly mentioned in the conversation.
2. Do NOT invent information that is not present in the text.
3. If no recipients are mentioned, return an empty array: []
4. Return ONLY the JSON array – no explanation, no extra text.

CONVERSATION:
{conversation}
"""
