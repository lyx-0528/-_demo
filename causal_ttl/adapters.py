from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import random

from .schema import Diagnosis, MedSample, ReasoningCluster, ReasoningPath
from .text_utils import answer_score, choose_distractor, normalize_text, split_sentences


class EdgeReasoner(ABC):
    @abstractmethod
    def generate_reasoning_paths(self, sample: MedSample, num_paths: int) -> list[ReasoningPath]:
        raise NotImplementedError

    @abstractmethod
    def revise_answer(
        self,
        sample: MedSample,
        path: ReasoningPath,
        diagnosis: Diagnosis | None,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def absorb_feedback(self, sample: MedSample, diagnosis: Diagnosis, strength: float) -> None:
        raise NotImplementedError


class CloudDiagnoser(ABC):
    @abstractmethod
    def diagnose(self, sample: MedSample, cluster: ReasoningCluster) -> Diagnosis:
        raise NotImplementedError


@dataclass(slots=True)
class MemoryRule:
    key_factor: str
    reasoning_rule: str
    corrected_answer: str
    strength: float = 0.0
    updates: int = 0


class MockEdgeReasoner(EdgeReasoner):
    def __init__(self, seed: int = 17) -> None:
        self.seed = seed
        self.memory: dict[str, MemoryRule] = {}
        self._generation_round = 0

    def generate_reasoning_paths(self, sample: MedSample, num_paths: int) -> list[ReasoningPath]:
        factor = self._extract_key_factor(sample)
        focus = sample.question.strip()[:24]
        memory_bonus = self._memory_relevance(sample)
        base_correct_rate = min(0.8, 0.25 + 0.35 * memory_bonus)
        paths: list[ReasoningPath] = []

        for index in range(num_paths):
            rng = random.Random(f"{self.seed}|{sample.sample_id}|{self._generation_round}|{index}")
            is_correct = rng.random() < (base_correct_rate + (0.12 if index == 0 else 0.0))
            error_type = "none" if is_correct else rng.choice(("logic", "factual", "omission", "shortcut"))
            answer = sample.gold_answer if is_correct else choose_distractor(sample.gold_answer, salt=index)
            score_hint = answer_score(sample.gold_answer, answer)
            text = self._build_reasoning_text(
                question_focus=focus,
                key_factor=factor,
                answer=answer,
                error_type=error_type,
            )
            paths.append(
                ReasoningPath(
                    path_id=f"{sample.sample_id}_path_{index}",
                    text=text,
                    answer=answer,
                    score_hint=score_hint,
                    metadata={
                        "is_correct": is_correct,
                        "error_type": error_type,
                        "memory_bonus": round(memory_bonus, 4),
                        "key_factor": factor,
                    },
                )
            )

        self._generation_round += 1
        return paths

    def revise_answer(
        self,
        sample: MedSample,
        path: ReasoningPath,
        diagnosis: Diagnosis | None,
    ) -> str:
        if diagnosis is None:
            return path.answer

        actual_error = str(path.metadata.get("error_type", "none"))
        memory_bonus = self._memory_relevance(sample)
        relevance = self._diagnosis_relevance(sample, diagnosis)
        answer_is_correct = answer_score(sample.gold_answer, diagnosis.corrected_answer) >= 0.95

        if answer_is_correct:
            match_bonus = 0.25 if diagnosis.error_type == actual_error or actual_error == "none" else -0.1
            intervention_strength = 0.45 + (0.25 * relevance) + (0.2 * memory_bonus) + match_bonus
            if intervention_strength >= 0.5:
                return sample.gold_answer
            return path.answer

        intervention_strength = 0.25 + (0.25 * relevance)
        if diagnosis.source == "swapped_control":
            intervention_strength -= 0.15
        if diagnosis.source == "corrupted_control":
            intervention_strength += 0.1
        if intervention_strength >= 0.45:
            return diagnosis.corrected_answer
        return path.answer

    def absorb_feedback(self, sample: MedSample, diagnosis: Diagnosis, strength: float) -> None:
        key = normalize_text(diagnosis.key_factor or diagnosis.corrected_answer or sample.gold_answer)
        if not key:
            key = normalize_text(sample.gold_answer)
        rule = self.memory.setdefault(
            key,
            MemoryRule(
                key_factor=diagnosis.key_factor or sample.gold_answer,
                reasoning_rule=diagnosis.reasoning_rule,
                corrected_answer=diagnosis.corrected_answer or sample.gold_answer,
            ),
        )
        rule.strength = min(2.0, rule.strength + max(0.0, strength))
        rule.updates += 1

    def _memory_relevance(self, sample: MedSample) -> float:
        evidence = normalize_text(f"{sample.question} {sample.knowledge} {sample.gold_answer}")
        strongest_match = 0.0
        for key, memory_rule in self.memory.items():
            if not key:
                continue
            if key in evidence or normalize_text(memory_rule.corrected_answer) == normalize_text(sample.gold_answer):
                strongest_match = max(strongest_match, memory_rule.strength)
        return min(1.0, strongest_match / 2.0)

    def _extract_key_factor(self, sample: MedSample) -> str:
        knowledge_sentences = split_sentences(sample.knowledge)
        if knowledge_sentences:
            return knowledge_sentences[0][:36]
        return sample.gold_answer

    def _diagnosis_relevance(self, sample: MedSample, diagnosis: Diagnosis) -> float:
        evidence = f"{sample.question} {sample.knowledge} {sample.gold_answer}"
        overlap = answer_score(evidence, diagnosis.key_factor or diagnosis.reasoning_rule)
        answer_overlap = answer_score(sample.gold_answer, diagnosis.corrected_answer)
        return min(1.0, max(overlap, answer_overlap))

    def _build_reasoning_text(
        self,
        question_focus: str,
        key_factor: str,
        answer: str,
        error_type: str,
    ) -> str:
        if error_type == "none":
            return (
                f"Step 1: Extract the decisive clue from the question: {question_focus}.\n"
                f"Step 2: Anchor the decision on the key medical factor: {key_factor}.\n"
                f"Step 3: Therefore the answer is: {answer}."
            )
        if error_type == "logic":
            return (
                f"Step 1: Notice a partial clue in the question: {question_focus}.\n"
                "Step 2: Jump to the conclusion before validating the causal bridge.\n"
                f"Step 3: Therefore the answer is: {answer}."
            )
        if error_type == "factual":
            return (
                f"Step 1: Focus on surface overlap in the question: {question_focus}.\n"
                "Step 2: Replace the decisive medical fact with an unsupported association.\n"
                f"Step 3: Therefore the answer is: {answer}."
            )
        if error_type == "omission":
            return (
                f"Step 1: Restate the symptoms from the question: {question_focus}.\n"
                "Step 2: Stop before identifying the decisive factor.\n"
                f"Step 3: Therefore the answer remains uncertain: {answer}."
            )
        return (
            f"Step 1: Match the question to a frequent template: {question_focus}.\n"
            "Step 2: Skip the explicit clue -> factor -> answer reasoning chain.\n"
            f"Step 3: Therefore the answer is: {answer}."
        )


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
