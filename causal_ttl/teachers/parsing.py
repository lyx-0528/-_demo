from __future__ import annotations

import json
import re
from typing import Any, Optional

from ..knowledge_tokens import KNOWLEDGE_TOKEN, normalize_knowledge_markers


_TOP_LEVEL_KEYS = ("causal_factors", "confounders", "steps", "facts", "counterfactuals", "answer")
_STEP_KEYS = ("goal", "causal", "needs_knowledge", "action", "query")
_COUNTERFACTUAL_KEYS = ("intervention", "expected_effect", "answer", "question")
_ALLOWED_ACTIONS = {"LOCAL_REASON", "ASK_CLOUD", "ANSWER"}


def _build_json_skeleton(*, arithmetic: bool) -> str:
    default_step = (
        '{"goal": "compute an intermediate quantity", "causal": "Compute the needed quantity to obtain '
        + KNOWLEDGE_TOKEN
        + '.", "needs_knowledge": true, "action": "ASK_CLOUD", "query": "What is the needed intermediate value?"}'
        if arithmetic
        else '{"goal": "identify the next causal sub-goal", "causal": "Use the relevant cause-effect relation to move one step closer to the answer.", "needs_knowledge": false, "action": "LOCAL_REASON", "query": ""}'
    )
    counterfactual = (
        '{"intervention": "change one causal factor", "expected_effect": "describe how the answer changes", '
        '"answer": "updated answer", "question": "What if the causal factor changed?"}'
    )
    answer = '"42"' if arithmetic else '"final answer"'
    facts = '["first computed value"]' if arithmetic else '["the needed external fact"]'
    return (
        "{\n"
        '  "causal_factors": [],\n'
        '  "confounders": [],\n'
        f'  "steps": [{default_step}],\n'
        f'  "facts": {facts},\n'
        f'  "counterfactuals": [{counterfactual}],\n'
        f'  "answer": {answer}\n'
        "}"
    )


def _build_schema_contract(*, arithmetic: bool) -> str:
    lines = [
        "Schema contract:",
        "1. Return a SINGLE JSON object and nothing else.",
        f"2. The top-level object must contain exactly these keys and no others: {', '.join(_TOP_LEVEL_KEYS)}.",
        f"3. Each item in steps must be an object with exactly these keys: {', '.join(_STEP_KEYS)}.",
        f"4. Each item in counterfactuals must be an object with exactly these keys: {', '.join(_COUNTERFACTUAL_KEYS)}.",
        f"5. In a step.causal string, replace any concrete missing fact or intermediate result with {KNOWLEDGE_TOKEN}.",
        '6. Allowed action values are exactly "LOCAL_REASON", "ASK_CLOUD", and "ANSWER".',
        '7. Use [] or "" instead of omitting a field.',
        '8. If action is not "ASK_CLOUD", query must be "".',
        '9. Do not add markdown fences, commentary, analysis, planning, or chain-of-thought outside the JSON object.',
        '10. Do not put nested JSON in any string field and do not use the legacy top-level key "causal_chain".',
    ]
    if arithmetic:
        lines.extend(
            [
                "11. For arithmetic or GSM8K problems, each intermediate numeric result needed later must appear as an ASK_CLOUD step.",
                "12. The number of facts must match the number of ASK_CLOUD steps, in order.",
                f"13. ASK_CLOUD queries must be concrete and self-contained; never place {KNOWLEDGE_TOKEN} inside query.",
                "14. answer must be a short numeric string for the final result.",
                "15. Use at least one ASK_CLOUD step when the problem needs multi-step arithmetic.",
            ]
        )
    return "\n".join(lines)


def _build_teacher_system_prompt(*, arithmetic: bool) -> str:
    role = (
        "You are a teacher that converts arithmetic word problems into a strict causal-intervention JSON plan."
        if arithmetic
        else "You are a causal-reasoning teacher that converts a question into structured causal supervision."
    )
    return (
        f"{role}\n"
        f"{_build_schema_contract(arithmetic=arithmetic)}\n"
        "Return only valid JSON.\n"
        "Use this exact JSON shape:\n"
        f"{_build_json_skeleton(arithmetic=arithmetic)}"
    )


def _build_repair_system_prompt(*, arithmetic: bool) -> str:
    task = "arithmetic/GSM8K drafts" if arithmetic else "causal reasoning drafts"
    return (
        f"You are a strict JSON normalizer for {task}. Convert the provided DRAFT into a SINGLE valid JSON object and nothing else.\n"
        f"{_build_schema_contract(arithmetic=arithmetic)}\n"
        'Preserve only information supported by the draft. If the draft does not support a field, use [] or "".\n'
        "Return only valid JSON."
    )


CAUSAL_TEACHER_SYSTEM_PROMPT = _build_teacher_system_prompt(arithmetic=False)
ARITHMETIC_CAUSAL_TEACHER_SYSTEM_PROMPT = _build_teacher_system_prompt(arithmetic=True)

KNOWLEDGE_FILL_SYSTEM_PROMPT = (
    "You are a cloud knowledge API invoked when an edge language model emits the special token "
    f"{KNOWLEDGE_TOKEN}. Given the QUESTION, the REASONING generated so far, and optionally the "
    "explicit QUERY, reply with only the concrete factual knowledge needed at that point. "
    "One or two concise sentences. No JSON."
)

CAUSAL_JSON_REPAIR_SYSTEM_PROMPT = _build_repair_system_prompt(arithmetic=False)
ARITHMETIC_CAUSAL_JSON_REPAIR_SYSTEM_PROMPT = _build_repair_system_prompt(arithmetic=True)

_ARITHMETIC_HINT_RE = re.compile(
    r"\b("
    r"how many|how much|total|left|remain|remaining|change|cost|price|each|per|every|"
    r"fraction|percent|times|sum|difference|product|quotient|more|less|altogether|"
    r"twice|half|quarter"
    r")\b",
    flags=re.IGNORECASE,
)
_DIGIT_RE = re.compile(r"\d")
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", flags=re.DOTALL | re.IGNORECASE)
_SCHEMA_ANCHOR_RE = re.compile(r'"(?:causal_factors|confounders|steps|facts|counterfactuals|answer)"')
_NUMERIC_VALUE_RE = re.compile(r"(?<![\w/])-?\$?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?")
_SECTION_LABEL_RE = re.compile(r"(?P<label>Causal factors|Confounders|Counterfactuals|Facts|Answer|Steps)\s*:")
_SECTION_LABEL_LINE_RE = re.compile(
    r"(?im)(?:^|\n)\s*(?P<label>causal factors|confounders|counterfactuals|facts|answer|steps)\s*:"
)
_STEP_LINE_RE = re.compile(
    r'(?im)^\s*(?:step\s*)?(?P<idx>\d+)[\.\):]?\s*goal\s*:\s*"?(?P<goal>[^"\n,]+)"?\s*,\s*'
    r'causal\s*:\s*"?(?P<causal>[^"\n]+)"?\s*,\s*'
    r'needs_knowledge\s*:\s*(?P<needs>true|false)\s*,\s*'
    r'action\s*:\s*(?P<action>LOCAL_REASON|ASK_CLOUD|ANSWER)\s*,\s*'
    r'query\s*:\s*"?(?P<query>[^"\n]*)"?\s*$'
)
_ARITHMETIC_PROSE_STEP_RE = re.compile(r"(?im)^\s*(?:step\s*)?(?P<idx>\d+)[\.\):]\s*(?P<body>.+)$")
_ARITHMETIC_RESULT_RE = re.compile(r"=\s*(?P<value>-?\$?\d+(?:\.\d+)?)")


def _extract_json_block(text: str) -> Optional[str]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0) if match else None


def _extract_balanced_json_block(text: str) -> Optional[str]:
    start = text.find("{")
    if start == -1:
        return None

    return _extract_balanced_json_block_from(text, start)


def _extract_balanced_json_block_from(text: str, start: int) -> Optional[str]:
    if start < 0 or start >= len(text) or text[start] != "{":
        return None

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _extract_all_balanced_json_blocks(text: str) -> list[str]:
    raw = text or ""
    blocks: list[str] = []
    start: Optional[int] = None
    depth = 0
    in_string = False
    escaped = False

    for index, char in enumerate(raw):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == "{":
            if depth == 0:
                start = index
            depth += 1
            continue

        if char == "}":
            if depth <= 0:
                continue
            depth -= 1
            if depth == 0 and start is not None:
                candidate = raw[start : index + 1].strip()
                if candidate:
                    blocks.append(candidate)
                start = None

    return blocks


def _extract_json_candidates(text: str) -> list[str]:
    raw = text.strip()
    candidates: list[str] = []

    def _append(candidate: Optional[str]) -> None:
        normalized = (candidate or "").strip()
        if normalized and normalized not in candidates:
            candidates.append(normalized)

    _append(raw)
    for match in _JSON_FENCE_RE.finditer(raw):
        _append(match.group(1))
    for match in _SCHEMA_ANCHOR_RE.finditer(raw):
        brace_index = raw.rfind("{", 0, match.start())
        _append(_extract_balanced_json_block_from(raw, brace_index))
    for candidate in reversed(_extract_all_balanced_json_blocks(raw)):
        _append(candidate)
    _append(_extract_balanced_json_block(raw))
    _append(_extract_json_block(raw))
    return candidates


def looks_like_arithmetic_question(question: str) -> bool:
    text = (question or "").strip()
    if not text:
        return False
    return bool(_DIGIT_RE.search(text) and _ARITHMETIC_HINT_RE.search(text))


def select_causal_teacher_system_prompt(question: str) -> str:
    return ARITHMETIC_CAUSAL_TEACHER_SYSTEM_PROMPT if looks_like_arithmetic_question(question) else CAUSAL_TEACHER_SYSTEM_PROMPT


def build_causal_teacher_user_prompt(question: str) -> str:
    stripped_question = (question or "").strip()
    arithmetic = looks_like_arithmetic_question(stripped_question)
    if arithmetic:
        return (
            f"QUESTION:\n{stripped_question}\n\n"
            "Return exactly one JSON object. The first character of your reply must be `{` and the last character must be `}`.\n"
            f"{_build_schema_contract(arithmetic=True)}\n"
            "Use this exact JSON skeleton:\n"
            f"{_build_json_skeleton(arithmetic=True)}"
        )
    return (
        f"QUESTION:\n{stripped_question}\n\n"
        "Return exactly one JSON object. The first character of your reply must be `{` and the last character must be `}`.\n"
        f"{_build_schema_contract(arithmetic=False)}\n"
        "Use this exact JSON skeleton:\n"
        f"{_build_json_skeleton(arithmetic=False)}"
    )


def select_causal_repair_system_prompt(question: str) -> str:
    return ARITHMETIC_CAUSAL_JSON_REPAIR_SYSTEM_PROMPT if looks_like_arithmetic_question(question) else CAUSAL_JSON_REPAIR_SYSTEM_PROMPT


def build_causal_repair_user_prompt(question: str, draft_response: str) -> str:
    arithmetic = looks_like_arithmetic_question(question)
    return (
        f"QUESTION:\n{(question or '').strip()}\n\n"
        "DRAFT RESPONSE TO NORMALIZE:\n"
        f"{(draft_response or '').strip()}\n\n"
        "Convert the draft into exactly one valid JSON object.\n"
        f"{_build_schema_contract(arithmetic=arithmetic)}\n"
        "Use this exact JSON skeleton:\n"
        f"{_build_json_skeleton(arithmetic=arithmetic)}"
    )


def _normalize_inline_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().strip("\"'")


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "0", "false", "no", "n", "none", "null"}:
            return False
        if normalized in {"1", "true", "yes", "y"}:
            return True
    return default


def _coerce_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        normalized = _normalize_inline_text(item)
        if normalized:
            items.append(normalized)
    return items


def _coerce_counterfactuals(value: Any) -> list[dict[str, str]]:
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []

    counterfactuals: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            intervention = _normalize_inline_text(item.get("intervention", ""))
            expected_effect = _normalize_inline_text(item.get("expected_effect", "") or item.get("expected_change", ""))
            answer = _clean_answer_candidate(
                str(item.get("answer", "") or item.get("counterfactual_answer", "") or "")
            )
            question = _normalize_inline_text(item.get("question", "") or item.get("counterfactual_question", ""))
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
                    "intervention": _normalize_inline_text(item),
                    "expected_effect": "",
                    "answer": "",
                    "question": "",
                }
            )
    return counterfactuals


def _normalize_action(raw_action: Any, *, needs_knowledge: bool, causal: str, query: str) -> str:
    action = _normalize_inline_text(raw_action).upper()
    if action in _ALLOWED_ACTIONS:
        return action
    if query or needs_knowledge or KNOWLEDGE_TOKEN in causal:
        return "ASK_CLOUD"
    if not causal:
        return "ANSWER"
    return "LOCAL_REASON"


def _coerce_step(step: Any) -> dict[str, Any]:
    if isinstance(step, str):
        step = {"causal": step}
    elif not isinstance(step, dict):
        step = {}

    goal = _normalize_inline_text(step.get("goal", "") or step.get("action_goal", ""))
    causal = normalize_knowledge_markers(_normalize_inline_text(step.get("causal", "") or step.get("reasoning", "")))
    query = _normalize_inline_text(step.get("query", "") or step.get("knowledge_query", ""))
    needs_knowledge = _coerce_bool(
        step.get("needs_knowledge", step.get("need_knowledge", None)),
        default=bool(query) or KNOWLEDGE_TOKEN in causal,
    )
    action = _normalize_action(step.get("action", ""), needs_knowledge=needs_knowledge, causal=causal, query=query)
    if action == "ASK_CLOUD":
        needs_knowledge = True
    else:
        query = ""
        if action == "ANSWER":
            needs_knowledge = False

    return {
        "goal": goal,
        "causal": causal,
        "needs_knowledge": needs_knowledge,
        "action": action,
        "query": query,
    }


def _clean_answer_candidate(candidate: str) -> str:
    cleaned = re.sub(r"\s+", " ", (candidate or "").strip()).strip(" .\n\r\t\"'")
    if not cleaned:
        return ""
    numbers = _NUMERIC_VALUE_RE.findall(cleaned)
    if numbers:
        return numbers[-1].strip()
    return cleaned


def _extract_answer_from_text(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""

    patterns = [
        r'(?im)^\s*"answer"\s*:\s*"?(?P<answer>[^"\n,}]+)',
        r"(?im)^(?!\s*counterfactual\s+answer\b)\s*answer\s*:\s*(?P<answer>[^\n]+)",
        r"(?im)^\s*final answer\s*:\s*(?P<answer>[^\n]+)",
        r'(?i)"answer"\s*:\s*"?(?P<answer>[^"\n,}]+)',
        r"(?i)\bthe\s+final\s+answer\s+is\s+(?P<answer>[^.\n]+)",
        r"(?i)\banswer\s+is\s+(?P<answer>[^.\n]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw)
        if match:
            normalized = _clean_answer_candidate(match.group("answer"))
            if normalized:
                return normalized
    return ""


def _extract_labeled_sections(text: str) -> dict[str, str]:
    raw = text or ""
    matches = list(_SECTION_LABEL_RE.finditer(raw))
    if not matches:
        matches = list(_SECTION_LABEL_LINE_RE.finditer(raw))
    if not matches:
        return {}

    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        key = str(match.group("label")).strip().lower().replace(" ", "_")
        value = raw[start:end].strip()
        if value:
            sections[key] = value
    return sections


def _clean_section_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).strip(" .\n\r\t\"'")


def _parse_section_items(text: str) -> list[str]:
    raw = _clean_section_text(text)
    if not raw or raw.lower().startswith("none"):
        return []
    items: list[str] = []
    for part in re.split(r"\s*,\s*", raw):
        normalized = _clean_section_text(re.sub(r"^(?:e\.g\.|for example)\s*,?\s*", "", part, flags=re.IGNORECASE))
        normalized = re.sub(r"\betc\.?$", "", normalized, flags=re.IGNORECASE).strip()
        if normalized and normalized.lower() not in {"none", "none obvious", "n/a"}:
            items.append(normalized)
    return items


def _parse_section_facts(text: str) -> list[str]:
    raw = _clean_section_text(text)
    if not raw or raw.lower().startswith("none"):
        return []
    return [value.strip() for value in _NUMERIC_VALUE_RE.findall(raw)]


def _parse_section_steps(text: str) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for line in (text or "").splitlines():
        match = _STEP_LINE_RE.match(line.strip())
        if not match:
            continue
        steps.append(
            _coerce_step(
                {
                    "goal": match.group("goal").strip(),
                    "causal": match.group("causal").strip(),
                    "needs_knowledge": match.group("needs").strip().lower() == "true",
                    "action": match.group("action").strip().upper(),
                    "query": match.group("query").strip(),
                }
            )
        )
    return steps


def _recover_structured_fields(text: str) -> dict[str, Any]:
    sections = _extract_labeled_sections(text)
    return {
        "causal_factors": _parse_section_items(sections.get("causal_factors", "")),
        "confounders": _parse_section_items(sections.get("confounders", "")),
        "counterfactuals": _coerce_counterfactuals(_parse_section_items(sections.get("counterfactuals", ""))),
        "facts": _parse_section_facts(sections.get("facts", "")),
        "steps": _parse_section_steps(sections.get("steps", "")),
    }


def _extract_arithmetic_step_result(body: str) -> str:
    matches = [match.group("value").strip() for match in _ARITHMETIC_RESULT_RE.finditer(body or "")]
    if matches:
        return matches[-1]
    numeric_values = [value.strip() for value in _NUMERIC_VALUE_RE.findall(body or "")]
    return numeric_values[-1] if numeric_values else ""


def _extract_arithmetic_step_expression(body: str) -> str:
    text = (body or "").strip()
    if "=" not in text:
        return ""
    parts = text.split("=")
    if len(parts) < 2:
        return ""
    candidate = parts[-2].strip()
    if ":" in candidate:
        candidate = candidate.split(":", 1)[-1].strip()
    candidate = re.sub(r"\s+", " ", candidate).strip(" .")
    return candidate


def _build_arithmetic_step_goal(body: str, index: int) -> str:
    text = (body or "").strip()
    if ":" in text:
        goal = text.split(":", 1)[0].strip()
        if goal:
            return goal
    return f"compute arithmetic step {index}"


def _build_arithmetic_step_causal(goal: str, is_final_step: bool) -> str:
    if is_final_step:
        return "Use the previously computed quantities to produce the final answer."
    goal_text = (goal or "the required intermediate quantity").strip().rstrip(".")
    goal_text = goal_text[0].lower() + goal_text[1:] if goal_text else "the required intermediate quantity"
    return f"Compute {goal_text} to obtain {KNOWLEDGE_TOKEN}."


def _build_arithmetic_step_query(body: str, fallback_goal: str) -> str:
    expression = _extract_arithmetic_step_expression(body)
    if expression:
        return f"What is {expression}?"
    fallback = (fallback_goal or "the required arithmetic quantity").strip().rstrip(".")
    return f"What is the result of {fallback}?"


def _recover_arithmetic_fields(text: str) -> dict[str, Any]:
    raw = text or ""
    if not looks_like_arithmetic_question(raw):
        return {"steps": [], "facts": [], "answer": ""}

    matches = list(_ARITHMETIC_PROSE_STEP_RE.finditer(raw))
    if not matches:
        return {"steps": [], "facts": [], "answer": ""}

    steps: list[dict[str, Any]] = []
    facts: list[str] = []
    answer = ""

    for position, match in enumerate(matches, start=1):
        body = (match.group("body") or "").strip()
        if not body:
            continue

        is_final_step = position == len(matches)
        goal = _build_arithmetic_step_goal(body, position)
        result_value = _extract_arithmetic_step_result(body)
        action = "ANSWER" if is_final_step else "ASK_CLOUD"
        step = {
            "goal": goal,
            "causal": _build_arithmetic_step_causal(goal, is_final_step),
            "needs_knowledge": not is_final_step,
            "action": action,
            "query": "" if is_final_step else _build_arithmetic_step_query(body, goal),
        }
        steps.append(step)

        if is_final_step:
            if result_value:
                answer = result_value
        elif result_value:
            facts.append(result_value)

    if not answer:
        answer = _extract_answer_from_text(raw)
    if not answer and facts:
        answer = facts[-1]

    return {"steps": steps, "facts": facts, "answer": answer}


def _empty_payload(causal_chain: str = "") -> dict[str, Any]:
    return {
        "causal_chain": causal_chain,
        "steps": [],
        "facts": [],
        "causal_factors": [],
        "confounders": [],
        "counterfactuals": [],
        "answer": "",
    }


def _coerce_facts(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    facts: list[str] = []
    for item in value:
        normalized = _normalize_inline_text(item)
        if normalized:
            facts.append(normalized)
    return facts


def _coerce_steps(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return []

    steps: list[dict[str, Any]] = []
    for item in value:
        step = _coerce_step(item)
        if step["goal"] or step["causal"] or step["query"] or step["action"] == "ANSWER":
            steps.append(step)
    return steps


def _coerce_causal_chain(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(_normalize_inline_text(item) for item in value if _normalize_inline_text(item))
    return _normalize_inline_text(value)


def _count_ask_cloud_steps(steps: Any) -> int:
    if not isinstance(steps, list):
        return 0
    return sum(1 for step in steps if isinstance(step, dict) and str(step.get("action", "")).upper() == "ASK_CLOUD")


def _step_is_complete(step: Any) -> bool:
    if not isinstance(step, dict):
        return False
    action = str(step.get("action", "")).strip().upper()
    if action not in _ALLOWED_ACTIONS:
        return False
    if action == "ASK_CLOUD":
        return bool(step.get("causal")) and bool(step.get("query"))
    if action == "ANSWER":
        return bool(step.get("goal") or step.get("causal"))
    return bool(step.get("causal"))


def _merge_missing_fields(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    merged = {
        "causal_chain": base.get("causal_chain", ""),
        "steps": list(base.get("steps", [])),
        "facts": list(base.get("facts", [])),
        "causal_factors": list(base.get("causal_factors", [])),
        "confounders": list(base.get("confounders", [])),
        "counterfactuals": list(base.get("counterfactuals", [])),
        "answer": str(base.get("answer", "") or ""),
    }
    if candidate.get("steps") and not merged["steps"]:
        merged["steps"] = list(candidate["steps"])
    if candidate.get("facts") and not merged["facts"]:
        merged["facts"] = list(candidate["facts"])
    if candidate.get("causal_factors") and not merged["causal_factors"]:
        merged["causal_factors"] = list(candidate["causal_factors"])
    if candidate.get("confounders") and not merged["confounders"]:
        merged["confounders"] = list(candidate["confounders"])
    if candidate.get("counterfactuals") and not merged["counterfactuals"]:
        merged["counterfactuals"] = list(candidate["counterfactuals"])
    if candidate.get("answer") and not merged["answer"]:
        merged["answer"] = str(candidate["answer"]).strip()
    if candidate.get("causal_chain") and not merged["causal_chain"]:
        merged["causal_chain"] = str(candidate["causal_chain"]).strip()
    return merged


def _iter_json_dicts(text: str) -> list[dict[str, Any]]:
    dictionaries: list[dict[str, Any]] = []
    for candidate in _extract_json_candidates(text or ""):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            dictionaries.append(parsed)
    return dictionaries


def _iter_payload_dict_candidates(payload: dict[str, Any], *, allow_nested: bool) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[int] = set()
    queue: list[dict[str, Any]] = [payload]

    while queue:
        current = queue.pop(0)
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        candidates.append(current)

        for key in ("payload", "result", "response", "data", "output"):
            nested = current.get(key)
            if isinstance(nested, dict):
                queue.append(nested)
            elif allow_nested and isinstance(nested, str) and "{" in nested:
                queue.extend(_iter_json_dicts(nested))

    return candidates


def _normalize_payload_dict(payload: dict[str, Any], *, raw_text: str, allow_nested: bool) -> dict[str, Any]:
    result = _empty_payload()
    result["causal_chain"] = _coerce_causal_chain(payload.get("causal_chain", "") or payload.get("reasoning", ""))
    result["steps"] = _coerce_steps(payload.get("steps", payload.get("causal_steps", payload.get("reasoning_steps", []))))
    result["facts"] = _coerce_facts(payload.get("facts", payload.get("knowledge", payload.get("observations", []))))
    result["causal_factors"] = _coerce_str_list(payload.get("causal_factors", payload.get("factors", [])))
    result["confounders"] = _coerce_str_list(payload.get("confounders", payload.get("spurious_correlations", [])))
    result["counterfactuals"] = _coerce_counterfactuals(
        payload.get("counterfactuals", payload.get("counterfactual_tests", []))
    )
    result["answer"] = _clean_answer_candidate(str(payload.get("answer", "") or payload.get("final_answer", "") or ""))
    if not result["answer"]:
        result["answer"] = _extract_answer_from_text(result["causal_chain"] or raw_text)

    if allow_nested:
        for key in ("causal_chain", "payload", "result", "response", "data", "output"):
            nested_text = payload.get(key)
            if not isinstance(nested_text, str) or "{" not in nested_text:
                continue
            nested_payload = parse_causal_payload(nested_text, _allow_nested=False)
            if score_causal_payload(nested_payload) > score_causal_payload(result):
                merged = _merge_missing_fields(nested_payload, result)
                if not merged.get("causal_chain"):
                    merged["causal_chain"] = result["causal_chain"]
                result = merged
            else:
                result = _merge_missing_fields(result, nested_payload)

    for source_text in (result["causal_chain"], raw_text):
        if not isinstance(source_text, str) or not source_text.strip():
            continue
        recovered = _recover_structured_fields(source_text)
        result = _merge_missing_fields(
            result,
            {
                "causal_chain": "",
                "steps": recovered.get("steps", []),
                "facts": recovered.get("facts", []),
                "causal_factors": recovered.get("causal_factors", []),
                "confounders": recovered.get("confounders", []),
                "counterfactuals": recovered.get("counterfactuals", []),
                "answer": _extract_answer_from_text(source_text),
            },
        )
        arithmetic_recovered = _recover_arithmetic_fields(source_text)
        result = _merge_missing_fields(
            result,
            {
                "causal_chain": "",
                "steps": arithmetic_recovered.get("steps", []),
                "facts": arithmetic_recovered.get("facts", []),
                "causal_factors": [],
                "confounders": [],
                "counterfactuals": [],
                "answer": arithmetic_recovered.get("answer", ""),
            },
        )

    if not result["causal_chain"] and result["steps"]:
        result["causal_chain"] = "\n".join(step["causal"] for step in result["steps"] if step.get("causal"))
    return result


def score_causal_payload(candidate: dict[str, Any]) -> tuple[int, int, int]:
    steps = candidate.get("steps", [])
    facts = candidate.get("facts", [])
    counterfactuals = candidate.get("counterfactuals", [])
    causal_factors = candidate.get("causal_factors", [])
    confounders = candidate.get("confounders", [])
    answer = str(candidate.get("answer", "") or "").strip()

    complete_steps = sum(1 for step in steps if _step_is_complete(step))
    ask_cloud_steps = _count_ask_cloud_steps(steps)
    score = 0
    if isinstance(steps, list) and steps:
        score += 8
    if answer:
        score += 6
    if isinstance(facts, list) and facts:
        score += 4
    if isinstance(counterfactuals, list) and counterfactuals:
        score += 3
    if isinstance(causal_factors, list) and causal_factors:
        score += 2
    if isinstance(confounders, list) and confounders:
        score += 1
    score += complete_steps
    if ask_cloud_steps == len(facts):
        score += 4
    elif ask_cloud_steps and facts:
        score += 1
    elif ask_cloud_steps == 0 and not facts:
        score += 1
    return score, complete_steps, len(json.dumps(candidate, ensure_ascii=False))


def needs_causal_payload_repair(payload: dict[str, Any]) -> bool:
    steps = payload.get("steps", [])
    facts = payload.get("facts", [])
    if not str(payload.get("answer", "") or "").strip():
        return True
    if not isinstance(steps, list) or not steps:
        return True
    if any(not _step_is_complete(step) for step in steps):
        return True
    ask_cloud_steps = _count_ask_cloud_steps(steps)
    if ask_cloud_steps != len(facts):
        return True
    return False


def parse_causal_payload(text: str, *, _allow_nested: bool = True) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return _empty_payload()

    best_payload: Optional[dict[str, Any]] = None
    best_score: Optional[tuple[int, int, int]] = None

    for candidate in _extract_json_candidates(raw):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(parsed, dict):
            continue

        for payload_candidate in _iter_payload_dict_candidates(parsed, allow_nested=_allow_nested):
            normalized = _normalize_payload_dict(payload_candidate, raw_text=raw, allow_nested=_allow_nested)
            score = score_causal_payload(normalized)
            if best_payload is None or best_score is None or score > best_score:
                best_payload = normalized
                best_score = score

    if best_payload is not None:
        return best_payload

    fallback = _empty_payload(raw)
    fallback["answer"] = _extract_answer_from_text(raw)
    recovered = _recover_structured_fields(raw)
    fallback = _merge_missing_fields(
        fallback,
        {
            "causal_chain": "",
            "steps": recovered.get("steps", []),
            "facts": recovered.get("facts", []),
            "causal_factors": recovered.get("causal_factors", []),
            "confounders": recovered.get("confounders", []),
            "counterfactuals": recovered.get("counterfactuals", []),
            "answer": "",
        },
    )
    arithmetic_recovered = _recover_arithmetic_fields(raw)
    fallback = _merge_missing_fields(
        fallback,
        {
            "causal_chain": "",
            "steps": arithmetic_recovered.get("steps", []),
            "facts": arithmetic_recovered.get("facts", []),
            "causal_factors": [],
            "confounders": [],
            "counterfactuals": [],
            "answer": arithmetic_recovered.get("answer", ""),
        },
    )
    return fallback
