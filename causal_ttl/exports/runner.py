from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schema import MedSample
from ..teachers import BaseTeacher
from .config import SupervisionExportConfig
from .records import build_supervision_records


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_supervision_dataset(
    samples: list[MedSample],
    teacher: BaseTeacher,
    output_dir: Path,
    config: SupervisionExportConfig,
) -> dict[str, Any]:
    records, payload_traces, summary, frontdoor_traces = build_supervision_records(
        samples,
        teacher,
        config,
        return_frontdoor_traces=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "supervision_dataset.jsonl", records)
    _write_jsonl(output_dir / "teacher_payloads.jsonl", payload_traces)
    if frontdoor_traces:
        _write_jsonl(output_dir / "frontdoor_traces.jsonl", frontdoor_traces)
    _write_jsonl(output_dir / "answer_records.jsonl", [record for record in records if record["task_type"] == "answer"])
    _write_jsonl(
        output_dir / "counterfactual_records.jsonl",
        [record for record in records if record["task_type"] == "counterfactual"],
    )
    _write_jsonl(output_dir / "policy_records.jsonl", [record for record in records if record["task_type"] == "policy"])
    (output_dir / "supervision_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
