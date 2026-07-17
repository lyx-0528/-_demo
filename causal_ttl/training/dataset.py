from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from ..data import load_jsonl


IGNORE_INDEX = -100
TASK_TYPE_IDS = {"answer": 0, "counterfactual": 1, "policy": 2}


class SupervisionDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def load_supervision_records(
    path: Path,
    *,
    task_types: Optional[Iterable[str]] = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    allowed = {task_type.strip() for task_type in task_types} if task_types else None
    rows = load_jsonl(path, limit=None)
    filtered: list[dict[str, Any]] = []
    for row in rows:
        task_type = str(row.get("task_type", "")).strip()
        if allowed is not None and task_type not in allowed:
            continue
        filtered.append(row)
        if limit is not None and len(filtered) >= limit:
            break
    return filtered


def _build_tokenized_example(
    prompt: str,
    target: str,
    tokenizer,
    max_length: int,
) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        target_ids = target_ids + [tokenizer.eos_token_id]

    if len(target_ids) >= max_length:
        target_ids = target_ids[:max_length]
        source_ids: list[int] = []
    else:
        max_source_len = max_length - len(target_ids)
        source_ids = prompt_ids[-max_source_len:]

    input_ids = source_ids + target_ids
    labels = [IGNORE_INDEX] * len(source_ids) + target_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def build_tokenized_supervision_dataset(
    records: list[dict[str, Any]],
    tokenizer,
    max_length: int,
) -> SupervisionDataset:
    rows: list[dict[str, Any]] = []
    for record in records:
        prompt = str(record.get("prompt", "")).strip()
        target = str(record.get("target", "")).strip()
        if not prompt or not target:
            continue

        task_type = str(record.get("task_type", "")).strip() or "answer"
        tokenized = _build_tokenized_example(prompt, target, tokenizer, max_length)
        tokenized["record_id"] = str(record.get("record_id", ""))
        tokenized["task_type"] = task_type
        tokenized["task_type_id"] = int(record.get("task_type_id", TASK_TYPE_IDS.get(task_type, 0)))
        tokenized["sample_weight"] = float(record.get("sample_weight", 1.0) or 1.0)
        rows.append(tokenized)
    return SupervisionDataset(rows)
