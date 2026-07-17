from __future__ import annotations

import json
import re
from typing import Any, Optional

from ..knowledge_tokens import KNOWLEDGE_TOKEN


CAUSAL_TEACHER_SYSTEM_PROMPT = (
    "You are a causal-reasoning teacher. For the given QUESTION, decompose the solution into an "
    "explicit causal reasoning PATH. Separate stable cause->effect logic from spurious correlations "
    "and specialized domain facts. Reply with a SINGLE JSON object and nothing else, using keys:\n"
    '  "causal_factors": strings naming the variables that genuinely change the answer\n'
    '  "confounders": strings naming background or lexical cues that correlate with the answer '
    "but should not be treated as causes\n"
    '  "steps": a list of objects, each with:\n'
    '      "goal"  - what this step tries to establish\n'
    '      "causal" - the transferable reasoning for this step. '
    f"Where a concrete fact, number, or entity is required, write {KNOWLEDGE_TOKEN} instead.\n"
    '      "needs_knowledge" - true if this step requires an external domain fact\n'
    '      "action" - one of "LOCAL_REASON", "ASK_CLOUD", "ANSWER"\n'
    '      "query" - if action is "ASK_CLOUD", the minimal knowledge question to ask the cloud\n'
    f'  "facts": strings giving concrete knowledge for each {KNOWLEDGE_TOKEN} in step order\n'
    '  "counterfactuals": a list of objects, each with "intervention", "expected_effect", '
    '"answer", and optional "question"\n'
    '  "answer": the final answer to the QUESTION\n'
    'You may also include legacy key "causal_chain", but prefer "steps". '
    "Return only valid JSON."
)

KNOWLEDGE_FILL_SYSTEM_PROMPT = (
    "You are a cloud knowledge API invoked when an edge language model emits the special token "
    f"{KNOWLEDGE_TOKEN}. Given the QUESTION, the REASONING generated so far, and optionally the "
    "explicit QUERY, reply with only the concrete factual knowledge needed at that point. "
    "One or two concise sentences. No JSON."
)


def _extract_json_block(text: str) -> Optional[str]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else None


def _coerce_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _coerce_counterfactuals(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []

    counterfactuals: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            intervention = str(item.get("intervention", "")).strip()
            expected_effect = str(item.get("expected_effect", "") or item.get("expected_change", "")).strip()
            answer = str(item.get("answer", "") or item.get("counterfactual_answer", "")).strip()
            question = str(item.get("question", "") or item.get("counterfactual_question", "")).strip()
            if intervention or expected_effect or answer or question:
                counterfactuals.append(
                    {
                        "intervention": intervention,
                        "expected_effect": expected_effect,
                        "answer": answer,
                        "question": question,
                    }
                )
        elif isinstance(item, str) and item.strip():
            counterfactuals.append(
                {
                    "intervention": item.strip(),
                    "expected_effect": "",
                    "answer": "",
                    "question": "",
                }
            )
    return counterfactuals


def _coerce_step(step: dict[str, Any]) -> dict[str, Any]:
    action = str(step.get("action", "") or "").strip().upper()
    if action not in {"LOCAL_REASON", "ASK_CLOUD", "ANSWER"}:
        action = "ASK_CLOUD" if step.get("needs_knowledge", False) else "LOCAL_REASON"

    causal = str(step.get("causal", "") or step.get("reasoning", "") or "").strip()
    return {
        "goal": str(step.get("goal", "") or step.get("action_goal", "") or "").strip(),
        "causal": causal,
        "needs_knowledge": bool(step.get("needs_knowledge", action == "ASK_CLOUD" or KNOWLEDGE_TOKEN in causal)),
        "action": action,
        "query": str(step.get("query", "") or step.get("knowledge_query", "") or "").strip(),
    }


def parse_causal_payload(text: str) -> dict[str, Any]:
    raw = text.strip()
    payload: Optional[dict[str, Any]] = None

    for candidate in (raw, _extract_json_block(raw)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            payload = parsed
            break

    if payload is None:
        return {
            "causal_chain": raw,
            "steps": [],
            "facts": [],
            "causal_factors": [],
            "confounders": [],
            "counterfactuals": [],
            "answer": "",
        }

    raw_chain = payload.get("causal_chain", "")
    if isinstance(raw_chain, list):
        causal_chain = "\n".join(str(step).strip() for step in raw_chain if str(step).strip())
    else:
        causal_chain = str(raw_chain or "").strip()

    facts = payload.get("facts", []) or []
    if isinstance(facts, str):
        facts = [facts]
    facts = [str(fact).strip() for fact in facts if str(fact).strip()]

    steps = payload.get("steps")
    if isinstance(steps, list):
        normalized_steps = []
        for step in steps:
            if isinstance(step, dict):
                normalized_steps.append(_coerce_step(step))
            elif isinstance(step, str) and step.strip():
                normalized_steps.append(
                    {
                        "goal": "",
                        "causal": step.strip(),
                        "needs_knowledge": KNOWLEDGE_TOKEN in step,
                        "action": "ASK_CLOUD" if KNOWLEDGE_TOKEN in step else "LOCAL_REASON",
                        "query": "",
                    }
                )
    else:
        normalized_steps = []

    return {
        "causal_chain": causal_chain,
        "steps": normalized_steps,
        "facts": facts,
        "causal_factors": _coerce_str_list(payload.get("causal_factors", [])),
        "confounders": _coerce_str_list(payload.get("confounders", [])),
        "counterfactuals": _coerce_counterfactuals(payload.get("counterfactuals", [])),
        "answer": str(payload.get("answer", "")).strip(),
    }
