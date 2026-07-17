from dataclasses import dataclass


@dataclass(slots=True)
class PipelineConfig:
    num_reasoning_paths: int = 8
    cluster_similarity_threshold: float = 0.45
    max_clusters: int = 4
    causal_margin: float = 0.15
    min_feedback_gain: float = 0.05
    seed: int = 17
