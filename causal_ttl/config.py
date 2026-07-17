from dataclasses import dataclass


@dataclass(slots=True)
class PipelineConfig:
    num_reasoning_paths: int = 8
    cluster_similarity_threshold: float = 0.45
    max_clusters: int = 4
    causal_margin: float = 0.15
    min_feedback_gain: float = 0.05
    seed: int = 17


@dataclass(slots=True)
class FrontDoorConfig:
    enabled: bool = False
    num_reasoning_paths: int = 8
    cluster_similarity_threshold: float = 0.45
    max_clusters: int = 4
    causal_margin: float = 0.15
    min_feedback_gain: float = 0.05
    answer_weight_floor: float = 1.0
    answer_weight_scale: float = 1.0
    counterfactual_weight_scale: float = 1.0
    policy_weight_scale: float = 1.0
    max_sample_weight: float = 4.0
    seed: int = 17
