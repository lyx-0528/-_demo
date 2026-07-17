from __future__ import annotations

from dataclasses import asdict
from statistics import mean
from typing import Any

from ..config import FrontDoorConfig, PipelineConfig
from ..pipeline import CausalTTLPipeline
from ..schema import Diagnosis, InterventionResult, MedSample


def build_pipeline_config(frontdoor: FrontDoorConfig) -> PipelineConfig:
    return PipelineConfig(
        num_reasoning_paths=frontdoor.num_reasoning_paths,
        cluster_similarity_threshold=frontdoor.cluster_similarity_threshold,
        max_clusters=frontdoor.max_clusters,
        causal_margin=frontdoor.causal_margin,
        min_feedback_gain=frontdoor.min_feedback_gain,
        seed=frontdoor.seed,
    )


def summarize_diagnosis(diagnosis: Diagnosis) -> dict[str, Any]:
    return {
        "cluster_id": diagnosis.cluster_id,
        "representative_path_id": diagnosis.representative_path_id,
        "error_type": diagnosis.error_type,
        "key_factor": diagnosis.key_factor,
        "reasoning_rule": diagnosis.reasoning_rule,
        "corrected_answer": diagnosis.corrected_answer,
        "confidence": diagnosis.confidence,
        "source": diagnosis.source,
    }


def summarize_intervention(intervention: InterventionResult) -> dict[str, Any]:
    return {
        "cluster_id": intervention.cluster_id,
        "cluster_weight": intervention.cluster_weight,
        "treated_score": intervention.treated_score,
        "null_score": intervention.null_score,
        "swapped_score": intervention.swapped_score,
        "corrupted_score": intervention.corrupted_score,
        "causal_gain": intervention.causal_gain,
        "weighted_gain": intervention.weighted_gain,
        "ttl_loss": intervention.ttl_loss,
        "diagnosis": summarize_diagnosis(intervention.diagnosis),
    }


def run_frontdoor_analysis(sample: MedSample, frontdoor: FrontDoorConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    pipeline = CausalTTLPipeline(config=build_pipeline_config(frontdoor))
    result = pipeline.run_sample(sample, apply_feedback=False)
    best_intervention = result.interventions[0] if result.interventions else None
    summary = {
        "baseline_answer": result.baseline_answer,
        "baseline_score": result.baseline_score,
        "final_answer": result.final_answer,
        "final_score": result.final_score,
        "difficulty": max(0.0, 1.0 - result.baseline_score),
        "chosen_cluster_id": result.chosen_cluster_id,
        "memory_update_strength": result.memory_update_strength,
        "cluster_count": len(result.clusters),
        "best_diagnosis": summarize_diagnosis(best_intervention.diagnosis) if best_intervention is not None else None,
        "best_intervention": summarize_intervention(best_intervention) if best_intervention is not None else None,
    }
    trace = {
        "sample": asdict(result.sample),
        "baseline_answer": result.baseline_answer,
        "baseline_score": result.baseline_score,
        "final_answer": result.final_answer,
        "final_score": result.final_score,
        "reasoning_paths": [asdict(path) for path in result.reasoning_paths],
        "clusters": [
            {
                "cluster_id": cluster.cluster_id,
                "weight": cluster.weight,
                "cohesion": cluster.cohesion,
                "representative": asdict(cluster.representative),
                "members": [asdict(member) for member in cluster.members],
            }
            for cluster in result.clusters
        ],
        "diagnoses": [summarize_diagnosis(diagnosis) for diagnosis in result.diagnoses],
        "interventions": [summarize_intervention(intervention) for intervention in result.interventions],
        "summary": summary,
    }
    return summary, trace


def compute_frontdoor_weight(
    task_type: str,
    frontdoor_summary: dict[str, Any] | None,
    frontdoor: FrontDoorConfig,
    *,
    has_facts: bool,
) -> float:
    if not frontdoor.enabled or frontdoor_summary is None:
        return 1.0

    best = frontdoor_summary.get("best_intervention") or {}
    difficulty = float(frontdoor_summary.get("difficulty", 0.0) or 0.0)
    causal_gain = max(0.0, float(best.get("weighted_gain", 0.0) or 0.0))
    ttl_loss = max(0.0, float(best.get("ttl_loss", 0.0) or 0.0))
    diagnosis = best.get("diagnosis") or {}
    needs_cloud = has_facts or str(diagnosis.get("error_type", "")).strip().lower() in {"factual", "omission"}

    if task_type == "answer":
        weight = frontdoor.answer_weight_floor + frontdoor.answer_weight_scale * (difficulty + causal_gain)
    elif task_type == "counterfactual":
        weight = 1.0 + frontdoor.counterfactual_weight_scale * (difficulty + ttl_loss)
    elif task_type == "policy":
        policy_signal = difficulty + causal_gain + (1.0 if needs_cloud else 0.0)
        weight = 1.0 + frontdoor.policy_weight_scale * policy_signal
    else:
        weight = 1.0

    return min(frontdoor.max_sample_weight, max(0.1, weight))


def flatten_frontdoor_summary(frontdoor_summary: dict[str, Any] | None) -> dict[str, Any]:
    if frontdoor_summary is None:
        return {}

    best = frontdoor_summary.get("best_intervention") or {}
    diagnosis = best.get("diagnosis") or {}
    return {
        "frontdoor_enabled": True,
        "frontdoor_baseline_score": float(frontdoor_summary.get("baseline_score", 0.0) or 0.0),
        "frontdoor_final_score": float(frontdoor_summary.get("final_score", 0.0) or 0.0),
        "frontdoor_difficulty": float(frontdoor_summary.get("difficulty", 0.0) or 0.0),
        "frontdoor_cluster_count": int(frontdoor_summary.get("cluster_count", 0) or 0),
        "frontdoor_best_cluster_id": str(best.get("cluster_id", "") or ""),
        "frontdoor_weighted_gain": float(best.get("weighted_gain", 0.0) or 0.0),
        "frontdoor_ttl_loss": float(best.get("ttl_loss", 0.0) or 0.0),
        "frontdoor_error_type": str(diagnosis.get("error_type", "") or ""),
        "frontdoor_key_factor": str(diagnosis.get("key_factor", "") or ""),
    }


def summarize_frontdoor_collection(frontdoor_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not frontdoor_summaries:
        return {"enabled": False, "num_samples": 0}

    gains = [float((summary.get("best_intervention") or {}).get("weighted_gain", 0.0) or 0.0) for summary in frontdoor_summaries]
    ttl_losses = [float((summary.get("best_intervention") or {}).get("ttl_loss", 0.0) or 0.0) for summary in frontdoor_summaries]
    baseline_scores = [float(summary.get("baseline_score", 0.0) or 0.0) for summary in frontdoor_summaries]
    final_scores = [float(summary.get("final_score", 0.0) or 0.0) for summary in frontdoor_summaries]
    difficulties = [float(summary.get("difficulty", 0.0) or 0.0) for summary in frontdoor_summaries]
    return {
        "enabled": True,
        "num_samples": len(frontdoor_summaries),
        "avg_baseline_score": mean(baseline_scores),
        "avg_final_score": mean(final_scores),
        "avg_difficulty": mean(difficulties),
        "avg_best_weighted_gain": mean(gains),
        "avg_best_ttl_loss": mean(ttl_losses),
        "positive_gain_rate": (sum(1 for gain in gains if gain > 0.0) / len(gains)) if gains else 0.0,
    }
