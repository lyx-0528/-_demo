from __future__ import annotations

from ..schema import Diagnosis, MedSample, ReasoningCluster
from ..text_utils import answer_score, split_sentences
from .interfaces import CloudDiagnoser


class HeuristicCloudDiagnoser(CloudDiagnoser):
    def diagnose(self, sample: MedSample, cluster: ReasoningCluster) -> Diagnosis:
        path = cluster.representative
        error_type = str(path.metadata.get("error_type", "none"))
        if answer_score(sample.gold_answer, path.answer) >= 0.95:
            error_type = "none"

        key_factor = self._extract_key_factor(sample)
        reasoning_rule = self._reasoning_rule(error_type, key_factor)
        confidence = min(0.98, 0.6 + (0.2 * cluster.cohesion) + (0.12 if error_type != "none" else 0.08))
        explanation = (
            f"The representative path in {cluster.cluster_id} is best corrected by focusing on "
            f"the decisive factor `{key_factor}` instead of the current {error_type} pattern."
        )
        return Diagnosis(
            error_type=error_type,
            key_factor=key_factor,
            reasoning_rule=reasoning_rule,
            corrected_answer=sample.gold_answer,
            explanation=explanation,
            confidence=confidence,
            representative_path_id=path.path_id,
            cluster_id=cluster.cluster_id,
            source="heuristic_cloud",
        )

    def _extract_key_factor(self, sample: MedSample) -> str:
        knowledge_sentences = split_sentences(sample.knowledge)
        if knowledge_sentences:
            return knowledge_sentences[0][:48]
        return sample.gold_answer

    def _reasoning_rule(self, error_type: str, key_factor: str) -> str:
        if error_type == "logic":
            return f"Preserve the causal chain from clue -> {key_factor} -> answer."
        if error_type == "factual":
            return f"Verify the decisive fact `{key_factor}` before mapping to an answer."
        if error_type == "omission":
            return f"Do not stop at symptom restatement; surface `{key_factor}` explicitly."
        if error_type == "shortcut":
            return f"Avoid direct pattern matching and route the reasoning through `{key_factor}`."
        return f"Keep the decisive factor `{key_factor}` stable when answering."
