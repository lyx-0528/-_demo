from __future__ import annotations

import json
from pathlib import Path

from .data import load_jsonl
from .text_utils import answer_score


def summarize_prediction_rows(rows: list[dict]) -> dict:
    scores = [answer_score(str(row.get("gold_answer", "")), str(row.get("predict", ""))) for row in rows]
    exact_matches = sum(1 for score in scores if score >= 0.95)
    unanswered = sum(1 for row in rows if not str(row.get("predict", "")).strip())
    cloud_calls = [int(row.get("cloud_calls", 0) or 0) for row in rows]
    return {
        "num_samples": len(rows),
        "avg_answer_score": (sum(scores) / len(scores)) if scores else 0.0,
        "exact_match": exact_matches,
        "exact_match_rate": (exact_matches / len(rows)) if rows else 0.0,
        "unanswered": unanswered,
        "avg_cloud_calls": (sum(cloud_calls) / len(cloud_calls)) if cloud_calls else 0.0,
    }


def evaluate_predictions_file(path: Path) -> dict:
    rows = load_jsonl(path)
    summary = summarize_prediction_rows(rows)
    output_path = path.with_name(path.stem + "_evaluation.json")
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
