from __future__ import annotations

import json
from pathlib import Path
import re

from .data import load_jsonl
from .text_utils import answer_score


def extract_gsm8k_answer_number(completion: str) -> str | None:
    tokens = re.split(r"[\s\n]+", completion or "")
    tokens_with_numbers = [token for token in tokens if re.search(r"-?\d", token)]
    cleaned_numbers = [re.sub(r"[^\d,\.-]", "", token) for token in tokens_with_numbers]

    if not cleaned_numbers:
        return None

    extracted_number = cleaned_numbers[-1].replace(",", "").strip(".")
    if not extracted_number:
        return None
    if extracted_number.count(".") > 1 or extracted_number.count("-") > 1:
        return None
    if not re.fullmatch(r"-?\d*\.?\d*", extracted_number):
        return None

    try:
        return str(int(round(float(extracted_number))))
    except ValueError:
        return None


def _row_gold_answer(row: dict) -> str:
    return str(row.get("gold_answer") or row.get("label") or "").strip()


def _resolve_metric(rows: list[dict], requested_metric: str) -> str:
    if requested_metric != "auto":
        return requested_metric
    if not rows:
        return "text"

    inspect_rows = rows[: min(len(rows), 50)]
    numeric_gold = sum(1 for row in inspect_rows if extract_gsm8k_answer_number(_row_gold_answer(row)) is not None)
    return "gsm8k" if numeric_gold >= max(1, int(len(inspect_rows) * 0.8)) else "text"


def summarize_prediction_rows(rows: list[dict], metric: str = "auto") -> dict:
    resolved_metric = _resolve_metric(rows, metric)
    unanswered = sum(1 for row in rows if not str(row.get("predict", "")).strip())
    cloud_calls = [int(row.get("cloud_calls", 0) or 0) for row in rows]

    if resolved_metric == "gsm8k":
        exact_matches = 0
        parsed_predictions = 0
        for row in rows:
            gold_number = extract_gsm8k_answer_number(_row_gold_answer(row))
            pred_number = extract_gsm8k_answer_number(str(row.get("predict", "")))
            if pred_number is not None:
                parsed_predictions += 1
            if gold_number is not None and pred_number is not None and gold_number == pred_number:
                exact_matches += 1

        return {
            "metric": "gsm8k_numeric_exact_match",
            "num_samples": len(rows),
            "exact_match": exact_matches,
            "exact_match_rate": (exact_matches / len(rows)) if rows else 0.0,
            "parsed_predictions": parsed_predictions,
            "parsed_prediction_rate": (parsed_predictions / len(rows)) if rows else 0.0,
            "unanswered": unanswered,
            "avg_cloud_calls": (sum(cloud_calls) / len(cloud_calls)) if cloud_calls else 0.0,
        }

    scores = [answer_score(_row_gold_answer(row), str(row.get("predict", ""))) for row in rows]
    exact_matches = sum(1 for score in scores if score >= 0.95)
    return {
        "metric": "text_similarity",
        "num_samples": len(rows),
        "avg_answer_score": (sum(scores) / len(scores)) if scores else 0.0,
        "exact_match": exact_matches,
        "exact_match_rate": (exact_matches / len(rows)) if rows else 0.0,
        "unanswered": unanswered,
        "avg_cloud_calls": (sum(cloud_calls) / len(cloud_calls)) if cloud_calls else 0.0,
    }


def evaluate_predictions_file(path: Path, metric: str = "auto") -> dict:
    rows = load_jsonl(path)
    summary = summarize_prediction_rows(rows, metric=metric)
    output_path = path.with_name(path.stem + "_evaluation.json")
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
