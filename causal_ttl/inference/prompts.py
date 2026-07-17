from __future__ import annotations

from typing import Any, Optional

from ..schema import MedSample


def _build_knowledge_block(facts: list[str]) -> str:
    lines = [str(fact).strip() for fact in facts if str(fact).strip()]
    if not lines:
        return ""
    return "[Cloud Knowledge]\n" + "\n".join(f"- {line}" for line in lines)


def build_inference_prompt(
    sample: MedSample,
    *,
    mode: str,
    teacher_payload: Optional[dict[str, Any]] = None,
    knowledge_at_inference: bool = True,
) -> str:
    normalized_mode = mode.strip().lower()
    question_block = f"Question:\n{sample.question.strip()}"
    facts = teacher_payload.get("facts", []) if teacher_payload else []

    if normalized_mode in {"fact", "v6"}:
        sections = [question_block]
        if knowledge_at_inference:
            knowledge_block = _build_knowledge_block(facts)
            if knowledge_block:
                sections.append(knowledge_block)
        return "\n\n".join(sections)

    if normalized_mode == "v7":
        return f"{question_block}\n\n[Reasoning]"

    raise ValueError(f"Unsupported inference mode `{mode}`. Supported modes: fact, v6, v7")
