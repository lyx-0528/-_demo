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
    "Do not output analysis, planning, or explanation outside the JSON object. "
    "Do not wrap the JSON in markdown fences. Return only valid JSON.\n"
    "Example JSON format:\n"
    "{\n"
    '  "causal_factors": ["factor A", "factor B"],\n'
    '  "confounders": ["surface cue"],\n'
    '  "steps": [\n'
    '    {"goal": "identify the missing fact", "causal": "The answer depends on <knowledge>; once known, continue causal reasoning.", "needs_knowledge": true, "action": "ASK_CLOUD", "query": "What fact is needed?"},\n'
    '    {"goal": "combine the fact with the question", "causal": "Use the fact to determine the answer.", "needs_knowledge": false, "action": "ANSWER", "query": ""}\n'
    "  ],\n"
    '  "facts": ["the needed fact"],\n'
    '  "counterfactuals": [{"intervention": "change factor A", "expected_effect": "the answer changes", "answer": "new answer", "question": "What if factor A changed?"}],\n'
    '  "answer": "final answer"\n'
    "}"
)

ARITHMETIC_CAUSAL_TEACHER_SYSTEM_PROMPT = (
    "You are a teacher that converts arithmetic word problems into a strict causal-intervention JSON plan. "
    "The QUESTION is a math or GSM8K-style reasoning problem. Reply with a SINGLE JSON object and nothing else.\n"
    "Required keys:\n"
    '  "causal_factors": strings naming the numeric quantities or variables that causally change the answer\n'
    '  "confounders": strings naming irrelevant story details or lexical distractions\n'
    '  "steps": a list of objects, each with:\n'
    '      "goal"  - the sub-goal of this arithmetic step\n'
    '      "causal" - abstract reasoning for the step. Replace every concrete intermediate number that will be computed '
    f'with {KNOWLEDGE_TOKEN}.\n'
    '      "needs_knowledge" - true when this step needs an intermediate numeric result that should be requested\n'
    '      "action" - one of "LOCAL_REASON", "ASK_CLOUD", "ANSWER"\n'
    '      "query" - for "ASK_CLOUD", ask for the exact intermediate computation, equation result, or unit conversion\n'
    f'  "facts": strings containing the concrete intermediate numeric results for each {KNOWLEDGE_TOKEN}, in step order\n'
    '  "counterfactuals": a list of objects, each with "intervention", "expected_effect", "answer", and optional "question"\n'
    '  "answer": the final numeric answer only, as a short string\n'
    "Rules:\n"
    "1. For arithmetic problems, every intermediate numeric result needed by a later step should appear as an ASK_CLOUD step.\n"
    "2. The number of facts must match the number of ASK_CLOUD observations in step order.\n"
    "3. Do not include prose before or after the JSON object.\n"
    "4. Do not wrap the JSON in markdown fences.\n"
    "5. Do not output analysis, self-correction, or chain-of-thought outside the JSON object.\n"
    "6. Use at least one ASK_CLOUD step whenever the problem requires multi-step arithmetic.\n"
    "Return only valid JSON.\n"
    "Example JSON format:\n"
    "{\n"
    '  "causal_factors": ["number of groups", "items per group"],\n'
    '  "confounders": ["irrelevant story details"],\n'
    '  "steps": [\n'
    '    {"goal": "compute the first intermediate quantity", "causal": "Multiply the relevant quantities to obtain <knowledge>.", "needs_knowledge": true, "action": "ASK_CLOUD", "query": "What is the product of the relevant quantities?"},\n'
    '    {"goal": "use the intermediate quantity to finish the problem", "causal": "Combine <knowledge> with the remaining quantities to reach the final answer.", "needs_knowledge": true, "action": "ASK_CLOUD", "query": "What is the final arithmetic result after combining the quantities?"},\n'
    '    {"goal": "state the final result", "causal": "Return the computed result as the final answer.", "needs_knowledge": false, "action": "ANSWER", "query": ""}\n'
    "  ],\n"
    '  "facts": ["first computed value", "final computed value"],\n'
    '  "counterfactuals": [{"intervention": "increase one quantity", "expected_effect": "the final answer increases", "answer": "updated numeric answer", "question": "What if one quantity were larger?"}],\n'
    '  "answer": "42"\n'
    "}"
)

KNOWLEDGE_FILL_SYSTEM_PROMPT = (
    "You are a cloud knowledge API invoked when an edge language model emits the special token "
    f"{KNOWLEDGE_TOKEN}. Given the QUESTION, the REASONING generated so far, and optionally the "
    "explicit QUERY, reply with only the concrete factual knowledge needed at that point. "
    "One or two concise sentences. No JSON."
)

CAUSAL_JSON_REPAIR_SYSTEM_PROMPT = (
    "You are a strict JSON normalizer. Convert the provided DRAFT into a SINGLE valid JSON object and nothing else. "
    "Use exactly these keys: causal_factors, confounders, steps, facts, counterfactuals, answer.\n"
    'Each step must be an object with keys: goal, causal, needs_knowledge, action, query.\n'
    'Allowed action values: "LOCAL_REASON", "ASK_CLOUD", "ANSWER".\n'
    "Preserve only information supported by the draft. If a field is missing, use [] or \"\". "
    "Do not include markdown fences or explanations. Return only valid JSON."
)

ARITHMETIC_CAUSAL_JSON_REPAIR_SYSTEM_PROMPT = (
    "You are a strict JSON normalizer for arithmetic/GSM8K drafts. Convert the provided DRAFT into a SINGLE valid JSON object and nothing else.\n"
    "Use exactly these keys: causal_factors, confounders, steps, facts, counterfactuals, answer.\n"
    'Each step must be an object with keys: goal, causal, needs_knowledge, action, query.\n'
    'Allowed action values: "LOCAL_REASON", "ASK_CLOUD", "ANSWER".\n'
    f'In causal fields, keep {KNOWLEDGE_TOKEN} placeholders when the draft describes intermediate values abstractly.\n'
    "Facts should contain concrete intermediate numeric results in step order. "
    "If the draft does not support a field, use [] or \"\". "
    "Do not include markdown fences or explanations. Return only valid JSON."
)

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
    if looks_like_arithmetic_question(stripped_question):
        return (
            f"QUESTION:\n{stripped_question}\n\n"
            "Return exactly one JSON object. The first character of your reply must be `{` and the last character must be `}`. "
            "Do not add explanations outside the JSON. For each intermediate arithmetic result needed later, create an ASK_CLOUD step "
            "and put the computed numeric result into facts in the same order.\n\n"
            "Use this JSON skeleton:\n"
            "{\n"
            '  "causal_factors": [],\n'
            '  "confounders": [],\n'
            '  "steps": [{"goal": "", "causal": "", "needs_knowledge": true, "action": "ASK_CLOUD", "query": ""}],\n'
            '  "facts": [],\n'
            '  "counterfactuals": [{"intervention": "", "expected_effect": "", "answer": "", "question": ""}],\n'
            '  "answer": ""\n'
            "}"
        )
    return (
        f"QUESTION:\n{stripped_question}\n\n"
        "Return exactly one JSON object. The first character of your reply must be `{` and the last character must be `}`. "
        "Do not add explanations outside the JSON.\n\n"
        "Use this JSON skeleton:\n"
        "{\n"
        '  "causal_factors": [],\n'
        '  "confounders": [],\n'
        '  "steps": [{"goal": "", "causal": "", "needs_knowledge": false, "action": "LOCAL_REASON", "query": ""}],\n'
        '  "facts": [],\n'
        '  "counterfactuals": [{"intervention": "", "expected_effect": "", "answer": "", "question": ""}],\n'
        '  "answer": ""\n'
        "}"
    )


def select_causal_repair_system_prompt(question: str) -> str:
    return ARITHMETIC_CAUSAL_JSON_REPAIR_SYSTEM_PROMPT if looks_like_arithmetic_question(question) else CAUSAL_JSON_REPAIR_SYSTEM_PROMPT


def build_causal_repair_user_prompt(question: str, draft_response: str) -> str:
    return (
        f"QUESTION:\n{(question or '').strip()}\n\n"
        "DRAFT RESPONSE TO NORMALIZE:\n"
        f"{(draft_response or '').strip()}\n\n"
        "Convert the draft into exactly one valid JSON object with keys "
        "causal_factors, confounders, steps, facts, counterfactuals, answer."
    )


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
            {
                "goal": match.group("goal").strip(),
                "causal": match.group("causal").strip(),
                "needs_knowledge": match.group("needs").strip().lower() == "true",
                "action": match.group("action").strip().upper(),
                "query": match.group("query").strip(),
            }
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


def score_causal_payload(candidate: dict[str, Any]) -> tuple[int, int]:
    steps = candidate.get("steps")
    facts = candidate.get("facts")
    counterfactuals = candidate.get("counterfactuals")
    causal_factors = candidate.get("causal_factors")
    confounders = candidate.get("confounders")
    answer = str(candidate.get("answer", "") or "").strip()

    score = 0
    if isinstance(steps, list) and steps:
        score += 5
    if isinstance(facts, list) and facts:
        score += 4
    if isinstance(counterfactuals, list) and counterfactuals:
        score += 3
    if isinstance(causal_factors, list) and causal_factors:
        score += 2
    if isinstance(confounders, list) and confounders:
        score += 1
    if answer:
        score += 2
    return score, len(json.dumps(candidate, ensure_ascii=False))


def needs_causal_payload_repair(payload: dict[str, Any]) -> bool:
    if not str(payload.get("answer", "") or "").strip():
        return True
    if not payload.get("steps"):
        return True
    if not payload.get("facts"):
        return True
    return False


def parse_causal_payload(text: str, *, _allow_nested: bool = True) -> dict[str, Any]:
    raw = text.strip()
    payload: Optional[dict[str, Any]] = None

    best_payload: Optional[dict[str, Any]] = None
    best_score: Optional[tuple[int, int]] = None

    for candidate in _extract_json_candidates(raw):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict):
            score = score_causal_payload(parsed)
            if best_payload is None or score > best_score:
                best_payload = parsed
                best_score = score

    payload = best_payload

    if payload is None:
        result = {
            "causal_chain": raw,
            "steps": [],
            "facts": [],
            "causal_factors": [],
            "confounders": [],
            "counterfactuals": [],
            "answer": _extract_answer_from_text(raw),
        }
        recovered = _recover_structured_fields(raw)
        for key in ("steps", "facts", "causal_factors", "confounders", "counterfactuals"):
            if recovered.get(key):
                result[key] = recovered[key]
        arithmetic_recovered = _recover_arithmetic_fields(raw)
        for key in ("steps", "facts"):
            if arithmetic_recovered.get(key) and not result.get(key):
                result[key] = arithmetic_recovered[key]
        if arithmetic_recovered.get("answer") and not result["answer"]:
            result["answer"] = arithmetic_recovered["answer"]
        return result

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

    result = {
        "causal_chain": causal_chain,
        "steps": normalized_steps,
        "facts": facts,
        "causal_factors": _coerce_str_list(payload.get("causal_factors", [])),
        "confounders": _coerce_str_list(payload.get("confounders", [])),
        "counterfactuals": _coerce_counterfactuals(payload.get("counterfactuals", [])),
        "answer": str(payload.get("answer", "")).strip() or _extract_answer_from_text(raw),
    }

    if _allow_nested and isinstance(causal_chain, str) and "{" in causal_chain:
        nested = parse_causal_payload(causal_chain, _allow_nested=False)
        if nested.get("steps") and not result["steps"]:
            result["steps"] = nested["steps"]
        if nested.get("facts") and not result["facts"]:
            result["facts"] = nested["facts"]
        if nested.get("causal_factors") and not result["causal_factors"]:
            result["causal_factors"] = nested["causal_factors"]
        if nested.get("confounders") and not result["confounders"]:
            result["confounders"] = nested["confounders"]
        if nested.get("counterfactuals") and not result["counterfactuals"]:
            result["counterfactuals"] = nested["counterfactuals"]
        if nested.get("answer") and not result["answer"]:
            result["answer"] = nested["answer"]

    for source_text in (causal_chain, raw):
        if not isinstance(source_text, str) or not source_text.strip():
            continue
        recovered = _recover_structured_fields(source_text)
        for key in ("steps", "facts", "causal_factors", "confounders", "counterfactuals"):
            if recovered.get(key) and not result.get(key):
                result[key] = recovered[key]
        arithmetic_recovered = _recover_arithmetic_fields(source_text)
        for key in ("steps", "facts"):
            if arithmetic_recovered.get(key) and not result.get(key):
                result[key] = arithmetic_recovered[key]
        if arithmetic_recovered.get("answer") and not result["answer"]:
            result["answer"] = arithmetic_recovered["answer"]

    return result
