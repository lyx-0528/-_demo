from __future__ import annotations

from ..config import PipelineConfig
from ..schema import MedSample, SampleRunResult
from ..text_utils import answer_score
from .clustering import cluster_reasoning_paths
from .diagnosers import HeuristicCloudDiagnoser
from .interventions import evaluate_interventions
from .reasoners import MockEdgeReasoner


class CausalTTLPipeline:
    def __init__(
        self,
        config: PipelineConfig | None = None,
        edge_reasoner=None,
        cloud_diagnoser=None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.edge_reasoner = edge_reasoner or MockEdgeReasoner(seed=self.config.seed)
        self.cloud_diagnoser = cloud_diagnoser or HeuristicCloudDiagnoser()

    def run_sample(self, sample: MedSample, *, apply_feedback: bool = True) -> SampleRunResult:
        reasoning_paths = self.edge_reasoner.generate_reasoning_paths(
            sample,
            num_paths=self.config.num_reasoning_paths,
        )
        clusters = cluster_reasoning_paths(
            reasoning_paths,
            similarity_threshold=self.config.cluster_similarity_threshold,
            max_clusters=self.config.max_clusters,
        )
        diagnoses = [self.cloud_diagnoser.diagnose(sample, cluster) for cluster in clusters]
        interventions = evaluate_interventions(
            sample,
            clusters=clusters,
            diagnoses=diagnoses,
            edge_reasoner=self.edge_reasoner,
            causal_margin=self.config.causal_margin,
        )

        baseline_cluster = max(clusters, key=lambda cluster: (cluster.weight, cluster.cohesion))
        baseline_answer = self.edge_reasoner.revise_answer(sample, baseline_cluster.representative, None)
        baseline_score = answer_score(sample.gold_answer, baseline_answer)

        final_answer = baseline_answer
        final_score = baseline_score
        chosen_cluster_id = ""
        memory_update_strength = 0.0

        if interventions and interventions[0].causal_gain >= self.config.min_feedback_gain:
            best_intervention = interventions[0]
            chosen_cluster_id = best_intervention.cluster_id
            memory_update_strength = max(0.0, best_intervention.weighted_gain)
            chosen_cluster = next(cluster for cluster in clusters if cluster.cluster_id == best_intervention.cluster_id)
            if apply_feedback:
                self.edge_reasoner.absorb_feedback(sample, best_intervention.diagnosis, memory_update_strength)
            final_answer = self.edge_reasoner.revise_answer(
                sample,
                chosen_cluster.representative,
                best_intervention.diagnosis,
            )
            final_score = answer_score(sample.gold_answer, final_answer)

        return SampleRunResult(
            sample=sample,
            baseline_answer=baseline_answer,
            baseline_score=baseline_score,
            final_answer=final_answer,
            final_score=final_score,
            reasoning_paths=reasoning_paths,
            clusters=clusters,
            diagnoses=diagnoses,
            interventions=interventions,
            chosen_cluster_id=chosen_cluster_id,
            memory_update_strength=memory_update_strength,
        )

    def run_dataset(self, samples: list[MedSample], *, apply_feedback: bool = True) -> list[SampleRunResult]:
        return [self.run_sample(sample, apply_feedback=apply_feedback) for sample in samples]
