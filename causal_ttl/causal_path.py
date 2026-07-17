from __future__ import annotations

import hashlib
import re
from typing import Any

from .knowledge_tokens import KNOWLEDGE_TOKEN, normalize_knowledge_markers


_STEP_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _coerce_steps(raw_steps: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_steps, list):
        return []
    steps: list[dict[str, Any]] = []
    for item in raw_steps:
        if isinstance(item, dict):
            steps.append(item)
        elif isinstance(item, str) and item.strip():
            steps.append({"goal": "", "causal": item.strip(), "needs_knowledge": KNOWLEDGE_TOKEN in item})
    return steps


def _legacy_chain_to_steps(causal_chain: Any) -> list[dict[str, Any]]:
    text = normalize_knowledge_markers(str(causal_chain or "").strip())
    if isinstance(causal_chain, list):
        parts = [normalize_knowledge_markers(str(item).strip()) for item in causal_chain if str(item).strip()]
    elif "\n" in text:
        parts = [part.strip() for part in text.split("\n") if part.strip()]
    else:
        parts = [part.strip() for part in _STEP_SPLIT_RE.split(text) if part.strip()]

    steps: list[dict[str, Any]] = []
    for index, part in enumerate(parts, start=1):
        has_knowledge = KNOWLEDGE_TOKEN in part
        steps.append(
            {
                "goal": f"Step {index}: move one causal hop closer to the answer",
                "causal": part,
                "needs_knowledge": has_knowledge,
                "action": "ASK_CLOUD" if has_knowledge else "LOCAL_REASON",
                "query": "",
            }
        )
    return steps


def extract_causal_steps(payload: dict[str, Any]) -> list[dict[str, Any]]:
    steps = _coerce_steps(payload.get("steps"))
    if steps:
        normalized_steps: list[dict[str, Any]] = []
        for step in steps:
            goal = str(step.get("goal", "") or step.get("action", "") or "").strip()
            causal = normalize_knowledge_markers(str(step.get("causal", "") or step.get("reasoning", "")).strip())
            needs_knowledge = step.get("needs_knowledge", step.get("need_knowledge", None))
            if needs_knowledge is None:
                needs_knowledge = KNOWLEDGE_TOKEN in causal or bool(step.get("fact"))
            if isinstance(needs_knowledge, str):
                needs_knowledge = needs_knowledge.strip().lower() not in ("none", "false", "no", "")
            action = str(step.get("action", "") or "").strip().upper()
            if action not in {"LOCAL_REASON", "ASK_CLOUD", "ANSWER"}:
                action = "ASK_CLOUD" if needs_knowledge else "LOCAL_REASON"
            normalized_steps.append(
                {
                    "goal": goal or "Advance the causal reasoning",
                    "causal": causal,
                    "needs_knowledge": bool(needs_knowledge),
                    "action": action,
                    "query": str(step.get("query", "") or step.get("knowledge_query", "") or "").strip(),
                }
            )
        return normalized_steps

    return _legacy_chain_to_steps(payload.get("causal_chain", ""))


def ensure_cloud_step(
    steps: list[dict[str, Any]],
    facts: list[str],
    payload: dict[str, Any],
    cloud_budget_ratio: float = 1.0,
    sample_key: str = "",
) -> list[dict[str, Any]]:
    if not facts:
        return steps

    has_ask = any(str(step.get("action", "")).strip().upper() == "ASK_CLOUD" for step in steps)
    if has_ask:
        return steps

    cloud_budget_ratio = max(0.0, min(1.0, float(cloud_budget_ratio)))
    if cloud_budget_ratio <= 0.0:
        return steps
    if cloud_budget_ratio < 1.0:
        key = sample_key or str(payload.get("answer", "")) or str(facts[0])
        bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        if bucket >= cloud_budget_ratio:
            return steps

    factors = payload.get("causal_factors", [])
    if isinstance(factors, str):
        factors = [factors]
    factor_hint = ", ".join(str(factor).strip() for factor in factors if str(factor).strip())
    query = "What external domain fact is needed to answer this question causally?"
    if factor_hint:
        query = f"What external domain fact is needed to decide these causal factors: {factor_hint}?"

    cloud_step = {
        "goal": "Acquire the external causal knowledge needed before local reasoning",
        "causal": (
            f"The answer depends on a domain fact {KNOWLEDGE_TOKEN}; once observed, "
            "the local model can continue the causal reasoning."
        ),
        "needs_knowledge": True,
        "action": "ASK_CLOUD",
        "query": query,
    }
    return [cloud_step] + steps


def _format_list_section(title: str, items: Any) -> str:
    if isinstance(items, str):
        items = [items]
    if not isinstance(items, list):
        return ""
    lines = [str(item).strip() for item in items if str(item).strip()]
    if not lines:
        return ""
    body = "\n".join(f"- {line}" for line in lines)
    return f"[{title}]\n{body}"


def format_causal_diagnostics(payload: dict[str, Any], facts: list[str], mask_facts_fn) -> str:
    sections = []

    factors = _format_list_section("Causal Factors", payload.get("causal_factors", []))
    if factors:
        sections.append(mask_facts_fn(factors, facts))

    confounders = _format_list_section("Confounders To Ignore", payload.get("confounders", []))
    if confounders:
        sections.append(mask_facts_fn(confounders, facts))

    counterfactuals = payload.get("counterfactuals", [])
    if isinstance(counterfactuals, list) and counterfactuals:
        lines = []
        for index, item in enumerate(counterfactuals, start=1):
            if isinstance(item, dict):
                intervention = str(item.get("intervention", "")).strip()
                effect = str(item.get("expected_effect", "") or item.get("expected_change", "")).strip()
                answer = str(item.get("answer", "") or item.get("counterfactual_answer", "")).strip()
            else:
                intervention = str(item).strip()
                effect = ""
                answer = ""

            parts = [f"CF{index} Intervention: {intervention}" if intervention else f"CF{index} Intervention:"]
            if effect:
                parts.append(f"Expected Effect: {effect}")
            if answer:
                parts.append(f"Counterfactual Answer: {answer}")
            lines.append("\n".join(parts))

        sections.append(mask_facts_fn("[Counterfactual Tests]\n" + "\n\n".join(lines), facts))

    return "\n\n".join(sections)


def format_causal_path(
    payload: dict[str, Any],
    facts: list[str],
    mask_facts_fn,
    cloud_budget_ratio: float = 1.0,
    sample_key: str = "",
) -> str:
    steps = ensure_cloud_step(
        extract_causal_steps(payload),
        facts,
        payload,
        cloud_budget_ratio=cloud_budget_ratio,
        sample_key=sample_key,
    )
    if not steps:
        chain = mask_facts_fn(normalize_knowledge_markers(str(payload.get("causal_chain", ""))), facts)
        return chain.strip()

    blocks: list[str] = []
    for index, step in enumerate(steps, start=1):
        goal = step["goal"].strip() or f"Step {index}: advance causal reasoning"
        causal = mask_facts_fn(step["causal"], facts).strip()
        has_knowledge = KNOWLEDGE_TOKEN in causal or step.get("needs_knowledge")
        action = str(step.get("action", "") or "").strip().upper()
        if action not in {"LOCAL_REASON", "ASK_CLOUD", "ANSWER"}:
            action = "ASK_CLOUD" if has_knowledge else "LOCAL_REASON"
        if not causal and action != "ASK_CLOUD":
            continue
        if not causal:
            causal = "Use the cloud observation only if it changes the causal decision."
        query = str(step.get("query", "") or "").strip()
        if action == "ASK_CLOUD" and not query:
            query = f"What specific knowledge is required to decide: {goal}"

        lines = [f"Step {index} | Goal: {goal}", f"Action: {action}"]
        if action == "ASK_CLOUD":
            lines.append(f"Query: {query}")
            lines.append(f"Observation: {KNOWLEDGE_TOKEN}")
        lines.append(f"Causal: {causal}")
        lines.append(f"Need: {KNOWLEDGE_TOKEN if has_knowledge else 'none'}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)
