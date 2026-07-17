from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from statistics import mean
import csv
import json


def summarize_results(results) -> dict:
    baseline_scores = [result.baseline_score for result in results]
    final_scores = [result.final_score for result in results]
    memory_updates = [result.memory_update_strength for result in results]
    causal_gains = [
        result.interventions[0].causal_gain
        for result in results
        if result.interventions
    ]
    improved = sum(1 for result in results if result.final_score > result.baseline_score)
    changed = sum(1 for result in results if result.final_answer != result.baseline_answer)
    return {
        "num_samples": len(results),
        "avg_baseline_score": mean(baseline_scores) if baseline_scores else 0.0,
        "avg_final_score": mean(final_scores) if final_scores else 0.0,
        "avg_score_gain": (mean(final_scores) - mean(baseline_scores)) if baseline_scores else 0.0,
        "improved_samples": improved,
        "changed_answers": changed,
        "avg_memory_update_strength": mean(memory_updates) if memory_updates else 0.0,
        "avg_best_causal_gain": mean(causal_gains) if causal_gains else 0.0,
    }


def flatten_result(result) -> dict:
    best = result.interventions[0] if result.interventions else None
    return {
        "sample_id": result.sample.sample_id,
        "source": result.sample.source,
        "question": result.sample.question,
        "gold_answer": result.sample.gold_answer,
        "baseline_answer": result.baseline_answer,
        "baseline_score": result.baseline_score,
        "final_answer": result.final_answer,
        "final_score": result.final_score,
        "chosen_cluster_id": result.chosen_cluster_id,
        "memory_update_strength": result.memory_update_strength,
        "best_diagnosis_error_type": best.diagnosis.error_type if best else "",
        "best_reasoning_rule": best.diagnosis.reasoning_rule if best else "",
        "best_causal_gain": best.causal_gain if best else 0.0,
        "best_weighted_gain": best.weighted_gain if best else 0.0,
        "best_ttl_loss": best.ttl_loss if best else 0.0,
    }


def write_json_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv_report(path: Path, results) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [flatten_result(result) for result in results]
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def serialize_results(results) -> list[dict]:
    return [asdict(result) for result in results]
