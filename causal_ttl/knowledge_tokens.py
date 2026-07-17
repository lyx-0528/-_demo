from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


KNOWLEDGE_TOKEN = "<knowledge>"
KNOWLEDGE_CLOSE = "</knowledge>"
KNOWLEDGE_SPECIAL_TOKENS = [KNOWLEDGE_TOKEN, KNOWLEDGE_CLOSE]
_LEGACY_SLOT_RE = re.compile(r"\[K\d+\]")


def find_unfilled_knowledge(text: str) -> Optional[re.Match[str]]:
    for match in re.finditer(re.escape(KNOWLEDGE_TOKEN), text):
        rest = text[match.end() :]
        next_open = rest.find(KNOWLEDGE_TOKEN)
        segment = rest if next_open == -1 else rest[:next_open]
        if KNOWLEDGE_CLOSE not in segment:
            return match
    return None


def has_unfilled_knowledge(text: str) -> bool:
    return find_unfilled_knowledge(text) is not None


def normalize_knowledge_markers(text: str) -> str:
    return _LEGACY_SLOT_RE.sub(KNOWLEDGE_TOKEN, text)


def register_knowledge_special_tokens(
    tokenizer: "PreTrainedTokenizer",
    model: Optional["PreTrainedModel"] = None,
) -> int:
    existing = set(tokenizer.get_vocab().keys())
    to_add = [token for token in KNOWLEDGE_SPECIAL_TOKENS if token not in existing]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": to_add})
    if model is not None and to_add:
        model.resize_token_embeddings(len(tokenizer))
    return len(to_add)


def wrap_filled_knowledge(fact: str) -> str:
    return f"{KNOWLEDGE_TOKEN} {fact.strip()} {KNOWLEDGE_CLOSE}"


def replace_first_unfilled_knowledge(text: str, fact: str) -> str:
    match = find_unfilled_knowledge(text)
    if not match:
        return text
    start, end = match.span()
    return text[:start] + wrap_filled_knowledge(fact) + text[end:]


def mask_leaked_facts_with_knowledge_token(text: str, facts: list[str]) -> str:
    masked_text = text
    for fact in facts:
        normalized_fact = (fact or "").strip()
        if len(normalized_fact) >= 4 and normalized_fact in masked_text:
            masked_text = masked_text.replace(normalized_fact, KNOWLEDGE_TOKEN)
    return masked_text
