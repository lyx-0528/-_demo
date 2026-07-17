from __future__ import annotations

import csv
import json
from pathlib import Path

from .schema import MedSample


ANSWER_PREFIXES = ("答案：", "答案:", "answer:", "answer：")
KNOWLEDGE_MARKERS = (
    "需要的医学知识包括",
    "以下是",
    "解决这个问题需要",
    "这题涉及到",
    "医学知识",
)


def _trim_answer(answer_text: str) -> str:
    for line in answer_text.replace("\r", "\n").split("\n"):
        line = line.strip()
        if not line:
            continue
        if line[0].isdigit() and "." in line[:3]:
            continue
        return line.strip("：:;；。 ")
    return answer_text.strip()


def _split_instruction_answer(assistant_text: str) -> tuple[str, str]:
    content = assistant_text.strip()
    lower_content = content.lower()
    extracted = content

    for prefix in ANSWER_PREFIXES:
        if prefix in content:
            extracted = content.split(prefix, 1)[1].strip()
            break
        if prefix in lower_content:
            start = lower_content.index(prefix)
            extracted = content[start + len(prefix) :].strip()
            break

    knowledge = ""
    for marker in KNOWLEDGE_MARKERS:
        if marker in extracted:
            answer_part, knowledge_part = extracted.split(marker, 1)
            return _trim_answer(answer_part), f"{marker}{knowledge_part}".strip()

    chunks = [chunk.strip() for chunk in extracted.split("\n\n") if chunk.strip()]
    if len(chunks) > 1:
        knowledge = "\n\n".join(chunks[1:])
    return _trim_answer(chunks[0] if chunks else extracted), knowledge


def load_medthink_csv(path: Path, limit: int | None = None) -> list[MedSample]:
    samples: list[MedSample] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            knowledge = (row.get("knowledge") or "").strip()
            if not question or not answer:
                continue
            samples.append(
                MedSample(
                    sample_id=f"{path.stem}-{index}",
                    question=question,
                    gold_answer=answer,
                    knowledge=knowledge,
                    source=path.name,
                )
            )
            if limit is not None and len(samples) >= limit:
                break
    return samples


def load_instruction_json(path: Path, limit: int | None = None) -> list[MedSample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples: list[MedSample] = []
    for index, entry in enumerate(payload):
        conversations = entry.get("conversations", [])
        if len(conversations) < 2:
            continue
        question = conversations[0].get("value", "").strip()
        assistant_text = conversations[1].get("value", "").strip()
        answer, knowledge = _split_instruction_answer(assistant_text)
        if not question or not answer:
            continue
        sample_id = entry.get("id") or f"{path.stem}-{index}"
        samples.append(
            MedSample(
                sample_id=sample_id,
                question=question,
                gold_answer=answer,
                knowledge=knowledge,
                source=path.name,
            )
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples


def load_prepared_json(path: Path, limit: int | None = None) -> list[MedSample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Prepared dataset {path} must be a JSON array.")

    samples: list[MedSample] = []
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict):
            continue
        question = str(entry.get("question", "")).strip()
        answer = str(entry.get("answer", "")).strip()
        knowledge = str(entry.get("knowledge", "")).strip()
        if not question or not answer:
            continue
        sample_id = str(entry.get("id") or f"{path.stem}-{index}")
        samples.append(
            MedSample(
                sample_id=sample_id,
                question=question,
                gold_answer=answer,
                knowledge=knowledge,
                source=str(entry.get("source") or path.name),
            )
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_samples_auto(path: Path, limit: int | None = None) -> list[MedSample]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_medthink_csv(path, limit=limit)
    if suffix == ".json":
        try:
            return load_prepared_json(path, limit=limit)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            return load_instruction_json(path, limit=limit)
    raise ValueError(f"Unsupported dataset format: {path}")
