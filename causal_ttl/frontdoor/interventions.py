from __future__ import annotations

from statistics import mean

from ..schema import Diagnosis, InterventionResult, MedSample, ReasoningCluster
from ..text_utils import answer_score, choose_distractor
from .interfaces import EdgeReasoner


def _build_corrupted_diagnosis(sample: MedSample, diagnosis: Diagnosis, salt: int) -> Diagnosis:
    corrupted_answer = choose_distractor(sample.gold_answer, salt=salt + 11)
    return Diagnosis(
        error_type="counterfactual_noise",
        key_factor="surface overlap",
        reasoning_rule="Overweight the first lexical match and ignore the decisive factor.",
        corrected_answer=corrupted_answer,
        explanation="Synthetic corrupted diagnosis used as a control intervention.",
        confidence=max(0.2, diagnosis.confidence * 0.5),
        representative_path_id=diagnosis.representative_path_id,
        cluster_id=diagnosis.cluster_id,
        source="corrupted_control",
    )


def _build_swapped_diagnosis(diagnosis: Diagnosis, diagnoses: list[Diagnosis]) -> Diagnosis:
    for other in diagnoses:
        if other.cluster_id != diagnosis.cluster_id and other.error_type != diagnosis.error_type:
            return Diagnosis(
                error_type=other.error_type,
                key_factor=other.key_factor,
                reasoning_rule=other.reasoning_rule,
                corrected_answer=other.corrected_answer,
                explanation="Diagnosis swapped from another cluster as a control intervention.",
                confidence=other.confidence,
                representative_path_id=diagnosis.representative_path_id,
                cluster_id=diagnosis.cluster_id,
                source="swapped_control",
            )
    return Diagnosis(
        error_type="swapped_control",
        key_factor="mismatched cluster rule",
        reasoning_rule="Apply a diagnosis that was derived from a different reasoning mode.",
        corrected_answer=diagnosis.corrected_answer,
        explanation="Fallback swapped diagnosis used when no alternative cluster exists.",
        confidence=max(0.2, diagnosis.confidence * 0.6),
        representative_path_id=diagnosis.representative_path_id,
        cluster_id=diagnosis.cluster_id,
        source="swapped_control",
    )


def evaluate_interventions(
    sample: MedSample,
    clusters: list[ReasoningCluster],
    diagnoses: list[Diagnosis],
    edge_reasoner: EdgeReasoner,
    causal_margin: float,
) -> list[InterventionResult]:
    diagnosis_by_id = {diagnosis.cluster_id: diagnosis for diagnosis in diagnoses}
    intervention_results: list[InterventionResult] = []

    for index, cluster in enumerate(clusters):
        diagnosis = diagnosis_by_id[cluster.cluster_id]
        swapped_diagnosis = _build_swapped_diagnosis(diagnosis, diagnoses)
        corrupted_diagnosis = _build_corrupted_diagnosis(sample, diagnosis, salt=index)

        null_answer = edge_reasoner.revise_answer(sample, cluster.representative, None)
        treated_answer = edge_reasoner.revise_answer(sample, cluster.representative, diagnosis)
        swapped_answer = edge_reasoner.revise_answer(sample, cluster.representative, swapped_diagnosis)
        corrupted_answer = edge_reasoner.revise_answer(sample, cluster.representative, corrupted_diagnosis)

        null_score = answer_score(sample.gold_answer, null_answer)
        treated_score = answer_score(sample.gold_answer, treated_answer)
        swapped_score = answer_score(sample.gold_answer, swapped_answer)
        corrupted_score = answer_score(sample.gold_answer, corrupted_answer)

        control_score = mean((null_score, swapped_score, corrupted_score))
        causal_gain = treated_score - control_score
        weighted_gain = cluster.weight * causal_gain
        ttl_loss = cluster.weight * ((1.0 - treated_score) + max(0.0, causal_margin - causal_gain))

        intervention_results.append(
            InterventionResult(
                cluster_id=cluster.cluster_id,
                diagnosis=diagnosis,
                cluster_weight=cluster.weight,
                treated_answer=treated_answer,
                treated_score=treated_score,
                null_answer=null_answer,
                null_score=null_score,
                swapped_answer=swapped_answer,
                swapped_score=swapped_score,
                corrupted_answer=corrupted_answer,
                corrupted_score=corrupted_score,
                causal_gain=causal_gain,
                weighted_gain=weighted_gain,
                ttl_loss=ttl_loss,
            )
        )

    return sorted(
        intervention_results,
        key=lambda result: (-result.weighted_gain, -result.treated_score, result.cluster_id),
    )
