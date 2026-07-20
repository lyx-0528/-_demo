from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from ..knowledge_tokens import KNOWLEDGE_TOKEN
from ..schema import MedSample
from .parsing import (
    KNOWLEDGE_FILL_SYSTEM_PROMPT,
    build_causal_teacher_user_prompt,
    parse_causal_payload,
    select_causal_teacher_system_prompt,
)


class BaseTeacher(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        raise NotImplementedError

    def generate_batch(self, prompts: list[str]) -> list[str]:
        return [self.generate(prompt) for prompt in prompts]

    def generate_causal(self, prompt: str) -> dict[str, Any]:
        teacher_prompt = build_causal_teacher_user_prompt(prompt)
        system_prompt = select_causal_teacher_system_prompt(prompt)
        return parse_causal_payload(self.generate(teacher_prompt, system_prompt=system_prompt))

    def generate_causal_batch(self, prompts: list[str]) -> list[dict[str, Any]]:
        return [self.generate_causal(prompt) for prompt in prompts]

    def generate_causal_sample(self, sample: MedSample) -> dict[str, Any]:
        return self.generate_causal(sample.question)

    def fill_knowledge(self, prompt: str, reasoning_prefix: str) -> str:
        user_prompt = (
            f"QUESTION:\n{prompt.strip()}\n\n"
            f"REASONING SO FAR:\n{reasoning_prefix.strip()}\n\n"
            f"Provide the knowledge to fill the next {KNOWLEDGE_TOKEN} marker:"
        )
        return self.generate(user_prompt, system_prompt=KNOWLEDGE_FILL_SYSTEM_PROMPT).strip()

    def fill_knowledge_sample(self, sample: MedSample, reasoning_prefix: str) -> str:
        return self.fill_knowledge(sample.question, reasoning_prefix)
