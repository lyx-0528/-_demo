from __future__ import annotations

from abc import ABC, abstractmethod

from ..schema import Diagnosis, MedSample, ReasoningCluster, ReasoningPath


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
