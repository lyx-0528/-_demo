from __future__ import annotations

import csv
import json
from pathlib import Path
import re

from .schema import MedSample


ANSWER_PREFIXES = (
    "\u7b54\u6848\uff1a",
    "\u7b54\u6848:",
    "answer:",
    "answer\uff1a",
)
KNOWLEDGE_MARKERS = (
    "\u9700\u8981\u7684\u533b\u5b66\u77e5\u8bc6\u5305\u62ec",
    "\u4ee5\u4e0b\u662f",
    "\u89e3\u51b3\u8fd9\u4e2a\u95ee\u9898\u9700\u8981",
    "\u8fd9\u9898\u6d89\u53ca\u5230",
    "\u533b\u5b66\u77e5\u8bc6",
)
INSTRUCTION_BLOCK_RE = re.compile(
    r"###\s*Instruction:\s*(.*?)(?:\n\s*###\s*Response:|\Z)",
    flags=re.IGNORECASE | re.DOTALL,
)


def _trim_answer(answer_text: str) -> str:
    for line in answer_text.replace("\r", "\n").split("\n"):
        line = line.strip()
        if not line:
            continue
        if re.match(r"^\d+\.", line):
            continue
        return line.strip(" :;,.!?")
    return answer_text.strip()


def _split_instruction_answer(assistant_text: str) -> tuple[str, str]:
    content = assistant_text.strip()
    lower_content = content.lower()
    extracted = content

    for prefix in ANSWER_PREFIXES:
        if prefix in content:
            extracted = content.split(prefix, 1)[1].strip()
            break
        lower_prefix = prefix.lower()
        if lower_prefix in lower_content:
            start = lower_content.index(lower_prefix)
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


def _extract_instruction_prompt(instruction_text: str, input_text: str = "") -> str:
    prompt = instruction_text.strip()
    match = INSTRUCTION_BLOCK_RE.search(prompt)
    if match:
        prompt = match.group(1).strip()

    input_text = input_text.strip()
    if input_text:
        prompt = f"{prompt}\n\n{input_text}"
    return prompt.strip()


def load_medthink_csv(path: Path, limit: int | None = None) -> list[MedSample]:
    samples: list[MedSample] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            question = str(row.get("question") or "").strip()
            answer = str(row.get("answer") or "").strip()
            knowledge = str(row.get("knowledge") or "").strip()
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
    if not isinstance(payload, list):
        raise ValueError(f"Instruction dataset {path} must be a JSON array.")

    samples: list[MedSample] = []
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict):
            continue

        sample_id = str(entry.get("id") or f"{path.stem}-{index}")
        knowledge = str(entry.get("knowledge") or "").strip()
        question = ""
        assistant_text = ""

        conversations = entry.get("conversations", [])
        if isinstance(conversations, list) and len(conversations) >= 2:
            question = str(conversations[0].get("value") or "").strip()
            assistant_text = str(conversations[1].get("value") or "").strip()
        else:
            instruction_text = str(entry.get("instruction") or "").strip()
            input_text = str(entry.get("input") or "").strip()
            assistant_text = str(entry.get("output") or "").strip()
            question = _extract_instruction_prompt(instruction_text, input_text)

        answer, parsed_knowledge = _split_instruction_answer(assistant_text)
        if not question or not answer:
            continue

        samples.append(
            MedSample(
                sample_id=sample_id,
                question=question,
                gold_answer=answer,
                knowledge=knowledge or parsed_knowledge,
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
            prepared_samples = load_prepared_json(path, limit=limit)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            prepared_samples = []
        if prepared_samples:
            return prepared_samples
        return load_instruction_json(path, limit=limit)
    raise ValueError(f"Unsupported dataset format: {path}")
