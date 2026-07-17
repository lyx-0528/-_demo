from __future__ import annotations

from dataclasses import field, dataclass
from typing import Any


@dataclass(slots=True)
class MedSample:
    sample_id: str
    question: str
    gold_answer: str
    knowledge: str = ""
    source: str = ""


@dataclass(slots=True)
class ReasoningPath:
    path_id: str
    text: str
    answer: str
    score_hint: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReasoningCluster:
    cluster_id: str
    representative: ReasoningPath
    members: list[ReasoningPath]
    weight: float
    cohesion: float


@dataclass(slots=True)
class Diagnosis:
    error_type: str
    key_factor: str
    reasoning_rule: str
    corrected_answer: str
    explanation: str
    confidence: float
    representative_path_id: str = ""
    cluster_id: str = ""
    source: str = "cloud"


@dataclass(slots=True)
class InterventionResult:
    cluster_id: str
    diagnosis: Diagnosis
    cluster_weight: float
    treated_answer: str
    treated_score: float
    null_answer: str
    null_score: float
    swapped_answer: str
    swapped_score: float
    corrupted_answer: str
    corrupted_score: float
    causal_gain: float
    weighted_gain: float
    ttl_loss: float


@dataclass(slots=True)
class SampleRunResult:
    sample: MedSample
    baseline_answer: str
    baseline_score: float
    final_answer: str
    final_score: float
    reasoning_paths: list[ReasoningPath]
    clusters: list[ReasoningCluster]
    diagnoses: list[Diagnosis]
    interventions: list[InterventionResult]
    chosen_cluster_id: str = ""
    memory_update_strength: float = 0.0
