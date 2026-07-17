from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ..causal_path import ensure_cloud_step, extract_causal_steps, format_causal_diagnostics, format_causal_path
from ..knowledge_tokens import KNOWLEDGE_TOKEN, mask_leaked_facts_with_knowledge_token, normalize_knowledge_markers
from ..schema import MedSample
from ..teachers import BaseTeacher
from .config import SupervisionExportConfig
from .frontdoor import (
    compute_frontdoor_weight,
    flatten_frontdoor_summary,
    run_frontdoor_analysis,
    summarize_frontdoor_collection,
)


TASK_TYPE_TO_ID = {"answer": 0, "counterfactual": 1, "policy": 2}


def _coerce_causal_chain(causal_chain: Any) -> str:
    if isinstance(causal_chain, list):
        return "\n".join(str(step).strip() for step in causal_chain if str(step).strip())
    return str(causal_chain or "").strip()


def _mask_facts_in_text(text: str, facts: list[str]) -> str:
    normalized_text = normalize_knowledge_markers(text)
    return mask_leaked_facts_with_knowledge_token(normalized_text, facts)


def _render_knowledge_block(facts: list[str]) -> str:
    lines = [str(fact).strip() for fact in facts if str(fact).strip()]
    if not lines:
        return ""
    return "[Cloud Knowledge]\n" + "\n".join(f"- {line}" for line in lines)


def _build_record(
    *,
    sample: MedSample,
    task_type: str,
    prompt: str,
    target: str,
    payload: dict[str, Any],
    facts: list[str],
    frontdoor_summary: dict[str, Any] | None = None,
    sample_weight: float = 1.0,
) -> dict[str, Any]:
    record = {
        "record_id": f"{sample.sample_id}:{task_type}",
        "sample_id": sample.sample_id,
        "task_type": task_type,
        "task_type_id": TASK_TYPE_TO_ID.get(task_type, 0),
        "source": sample.source,
        "question": sample.question,
        "gold_answer": sample.gold_answer,
        "teacher_answer": str(payload.get("answer", "")).strip(),
        "prompt": prompt,
        "target": target,
        "sample_weight": float(sample_weight),
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target},
        ],
        "facts": facts,
        "causal_factors": payload.get("causal_factors", []),
        "confounders": payload.get("confounders", []),
    }
    record.update(flatten_frontdoor_summary(frontdoor_summary))
    return record


def _build_answer_record(
    sample: MedSample,
    payload: dict[str, Any],
    facts: list[str],
    config: SupervisionExportConfig,
    frontdoor_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt_sections = [f"Question:\n{sample.question.strip()}"]
    if config.knowledge_in_context:
        knowledge_block = _render_knowledge_block(facts)
        if knowledge_block:
            prompt_sections.append(knowledge_block)
    elif config.supervise_reasoning:
        prompt_sections.append("[Reasoning]")
    prompt = "\n\n".join(prompt_sections)

    if config.supervise_reasoning:
        diagnostics_text = format_causal_diagnostics(payload, facts, _mask_facts_in_text)
        if config.causal_path_format:
            reasoning_text = format_causal_path(
                payload,
                facts,
                _mask_facts_in_text,
                cloud_budget_ratio=config.cloud_budget_ratio,
                sample_key=sample.question,
            )
        else:
            reasoning_text = _mask_facts_in_text(_coerce_causal_chain(payload.get("causal_chain", "")), facts)
        if diagnostics_text:
            reasoning_text = f"{diagnostics_text}\n\n[Reasoning Path]\n{reasoning_text}".strip()
    else:
        reasoning_text = ""

    target = f"{reasoning_text}\n\nAnswer: {sample.gold_answer}".strip() if reasoning_text else f"Answer: {sample.gold_answer}"
    sample_weight = compute_frontdoor_weight(
        "answer",
        frontdoor_summary,
        config.frontdoor,
        has_facts=bool(facts),
    )
    return _build_record(
        sample=sample,
        task_type="answer",
        prompt=prompt,
        target=target,
        payload=payload,
        facts=facts,
        frontdoor_summary=frontdoor_summary,
        sample_weight=sample_weight,
    )


def _build_policy_record(
    sample: MedSample,
    payload: dict[str, Any],
    facts: list[str],
    config: SupervisionExportConfig,
    frontdoor_summary: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    steps = ensure_cloud_step(
        extract_causal_steps(payload),
        facts,
        payload,
        cloud_budget_ratio=config.cloud_budget_ratio,
        sample_key=sample.question,
    )
    policy_lines: list[str] = []
    if steps:
        for index, step in enumerate(steps, start=1):
            action = str(step.get("action", "") or "").strip().upper()
            needs_knowledge = bool(step.get("needs_knowledge", False))
            causal = str(step.get("causal", "") or step.get("reasoning", "") or "").strip()
            if action not in {"LOCAL_REASON", "ASK_CLOUD", "ANSWER"}:
                action = "ASK_CLOUD" if needs_knowledge or KNOWLEDGE_TOKEN in causal else "LOCAL_REASON"
            query = str(step.get("query", "") or step.get("knowledge_query", "") or "").strip()
            policy_lines.append(f"Step {index} Action: {action}")
            if action == "ASK_CLOUD":
                if not query:
                    goal = str(step.get("goal", "") or "the current causal decision").strip()
                    query = f"What knowledge is required for {goal}?"
                policy_lines.append(f"Step {index} Query: {query}")
    elif facts:
        policy_lines.extend(["Step 1 Action: LOCAL_REASON", "Step 2 Action: ANSWER"])

    if not policy_lines:
        return None

    prompt = (
        f"Question:\n{sample.question.strip()}\n\n"
        "[Cloud Action Policy]\n"
        "Decide when the local model should reason locally or ask the cloud."
    )
    target = "\n".join(policy_lines)
    sample_weight = compute_frontdoor_weight(
        "policy",
        frontdoor_summary,
        config.frontdoor,
        has_facts=bool(facts),
    )
    return _build_record(
        sample=sample,
        task_type="policy",
        prompt=prompt,
        target=target,
        payload=payload,
        facts=facts,
        frontdoor_summary=frontdoor_summary,
        sample_weight=sample_weight,
    )


def _build_counterfactual_record(
    sample: MedSample,
    payload: dict[str, Any],
    counterfactual: dict[str, Any],
    facts: list[str],
    config: SupervisionExportConfig,
    frontdoor_summary: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    intervention = str(counterfactual.get("intervention", "")).strip()
    question = str(counterfactual.get("question", "") or counterfactual.get("counterfactual_question", "")).strip()
    expected_effect = str(
        counterfactual.get("expected_effect", "") or counterfactual.get("expected_change", "")
    ).strip()
    answer = str(counterfactual.get("answer", "") or counterfactual.get("counterfactual_answer", "")).strip()
    if not (intervention or question or expected_effect or answer):
        return None

    prompt_parts = [f"Question:\n{sample.question.strip()}", "[Counterfactual Intervention]"]
    if intervention:
        prompt_parts.append(f"do({intervention})")
    if question:
        prompt_parts.append(f"Counterfactual Question: {question}")
    prompt_parts.append("[Reasoning]")
    prompt = "\n\n".join(prompt_parts)

    target_parts = []
    if expected_effect:
        target_parts.append(f"Expected Effect: {expected_effect}")
    if answer:
        target_parts.append(f"Answer: {answer}")
    if not target_parts:
        target_parts.append("Expected Effect: unchanged unless the intervention changes a causal factor.")
    target = "\n".join(target_parts)
    sample_weight = compute_frontdoor_weight(
        "counterfactual",
        frontdoor_summary,
        config.frontdoor,
        has_facts=bool(facts),
    )
    return _build_record(
        sample=sample,
        task_type="counterfactual",
        prompt=prompt,
        target=target,
        payload=payload,
        facts=facts,
        frontdoor_summary=frontdoor_summary,
        sample_weight=sample_weight,
    )


def _build_fallback_counterfactual_record(
    sample: MedSample,
    payload: dict[str, Any],
    facts: list[str],
    config: SupervisionExportConfig,
    frontdoor_summary: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if not facts:
        return None
    prompt = (
        f"Question:\n{sample.question.strip()}\n\n"
        "[Counterfactual Intervention]\n"
        "do(remove_or_change_cloud_evidence)\n\n"
        "Counterfactual Question: Would the final answer remain valid if the cloud evidence were false or unavailable?\n\n"
        "[Reasoning]"
    )
    target = (
        "Expected Effect: Re-check the answer against causal evidence; change it only when the removed evidence "
        "is necessary for the causal decision.\n"
        f"Answer: {sample.gold_answer}"
    )
    sample_weight = compute_frontdoor_weight(
        "counterfactual",
        frontdoor_summary,
        config.frontdoor,
        has_facts=bool(facts),
    )
    return _build_record(
        sample=sample,
        task_type="counterfactual",
        prompt=prompt,
        target=target,
        payload=payload,
        facts=facts,
        frontdoor_summary=frontdoor_summary,
        sample_weight=sample_weight,
    )


def build_supervision_records(
    samples: list[MedSample],
    teacher: BaseTeacher,
    config: SupervisionExportConfig,
    *,
    return_frontdoor_traces: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]] | tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    records: list[dict[str, Any]] = []
    payload_traces: list[dict[str, Any]] = []
    frontdoor_traces: list[dict[str, Any]] = []
    frontdoor_summaries: list[dict[str, Any]] = []
    counts = {"answer": 0, "counterfactual": 0, "policy": 0}

    for sample in samples:
        payload = teacher.generate_causal_sample(sample)
        facts = [str(fact).strip() for fact in payload.get("facts", []) if str(fact).strip()]
        frontdoor_summary = None
        if config.frontdoor.enabled:
            frontdoor_summary, frontdoor_trace = run_frontdoor_analysis(sample, config.frontdoor)
            frontdoor_traces.append(frontdoor_trace)
            frontdoor_summaries.append(frontdoor_summary)
        formatted_path = format_causal_path(
            payload,
            facts,
            _mask_facts_in_text,
            cloud_budget_ratio=config.cloud_budget_ratio,
            sample_key=sample.question,
        )
        payload_traces.append(
            {
                "sample_id": sample.sample_id,
                "source": sample.source,
                "question": sample.question,
                "gold_answer": sample.gold_answer,
                "payload": payload,
                "formatted_path": formatted_path,
                "frontdoor_summary": frontdoor_summary,
            }
        )

        records.append(_build_answer_record(sample, payload, facts, config, frontdoor_summary=frontdoor_summary))
        counts["answer"] += 1

        if not config.include_causal_loss_samples:
            continue

        policy_record = _build_policy_record(sample, payload, facts, config, frontdoor_summary=frontdoor_summary)
        if policy_record is not None:
            records.append(policy_record)
            counts["policy"] += 1

        added_counterfactual = False
        for counterfactual in payload.get("counterfactuals", []):
            if not isinstance(counterfactual, dict):
                continue
            counterfactual_record = _build_counterfactual_record(
                sample,
                payload,
                counterfactual,
                facts,
                config,
                frontdoor_summary=frontdoor_summary,
            )
            if counterfactual_record is not None:
                records.append(counterfactual_record)
                counts["counterfactual"] += 1
                added_counterfactual = True
        if not added_counterfactual:
            fallback_record = _build_fallback_counterfactual_record(
                sample,
                payload,
                facts,
                config,
                frontdoor_summary=frontdoor_summary,
            )
            if fallback_record is not None:
                records.append(fallback_record)
                counts["counterfactual"] += 1

    summary = {
        "num_samples": len(samples),
        "num_records": len(records),
        "counts": counts,
        "config": asdict(config),
        "frontdoor": summarize_frontdoor_collection(frontdoor_summaries),
    }
    if return_frontdoor_traces:
        return records, payload_traces, summary, frontdoor_traces
    return records, payload_traces, summary
