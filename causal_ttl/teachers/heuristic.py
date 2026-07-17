from __future__ import annotations

import json
from typing import Any, Optional

from ..knowledge_tokens import KNOWLEDGE_TOKEN
from ..schema import MedSample
from ..text_utils import split_sentences
from .base import BaseTeacher
from .cache import TeacherCache


class HeuristicCausalTeacher(BaseTeacher):
    def __init__(self, cache_dir: Optional[str] = None) -> None:
        self.cache = TeacherCache(cache_dir)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        key = json.dumps({"prompt": prompt, "system_prompt": system_prompt, "backend": "heuristic"}, ensure_ascii=False)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

        payload = {
            "causal_factors": ["the decisive clue in the question"],
            "confounders": ["surface lexical overlap without causal support"],
            "steps": [
                {
                    "goal": "Locate the decisive clue in the question",
                    "causal": "Identify the symptom, mechanism, or concept that causally determines the answer.",
                    "needs_knowledge": False,
                    "action": "LOCAL_REASON",
                    "query": "",
                },
                {
                    "goal": "Request the missing domain fact only if it is necessary",
                    "causal": f"Use {KNOWLEDGE_TOKEN} only when the answer depends on specialized domain knowledge.",
                    "needs_knowledge": True,
                    "action": "ASK_CLOUD",
                    "query": "What domain fact is needed to resolve the decisive clue?",
                },
                {
                    "goal": "Map the causal evidence to the final answer",
                    "causal": "Once the decisive evidence is established, choose the answer that follows from it.",
                    "needs_knowledge": False,
                    "action": "ANSWER",
                    "query": "",
                },
            ],
            "facts": [],
            "counterfactuals": [
                {
                    "intervention": "remove the decisive clue from the case",
                    "expected_effect": "The answer should be re-evaluated because the causal basis changes.",
                    "answer": "",
                    "question": "",
                }
            ],
            "answer": "",
        }
        response = json.dumps(payload, ensure_ascii=False)
        self.cache.set(key, prompt, response, {"backend": "heuristic"})
        return response

    def generate_causal_sample(self, sample: MedSample) -> dict[str, Any]:
        facts = split_sentences(sample.knowledge)
        decisive_fact = facts[0] if facts else ""
        factor = decisive_fact[:64] if decisive_fact else sample.gold_answer

        steps = [
            {
                "goal": "Identify the decisive medical clue",
                "causal": "Extract the symptom, condition, or mechanism that determines the answer.",
                "needs_knowledge": False,
                "action": "LOCAL_REASON",
                "query": "",
            }
        ]
        if decisive_fact:
            steps.append(
                {
                    "goal": "Retrieve the external medical fact needed for the decision",
                    "causal": f"The answer depends on the domain fact {KNOWLEDGE_TOKEN}.",
                    "needs_knowledge": True,
                    "action": "ASK_CLOUD",
                    "query": "What medical fact resolves the decisive clue in this question?",
                }
            )
        steps.append(
            {
                "goal": "Map the causal evidence to the final answer",
                "causal": "Use the established evidence to select the answer that best follows from it.",
                "needs_knowledge": False,
                "action": "ANSWER",
                "query": "",
            }
        )

        return {
            "causal_chain": "\n".join(step["causal"] for step in steps),
            "steps": steps,
            "facts": facts[:1],
            "causal_factors": [factor],
            "confounders": ["surface wording overlap without verifying the decisive factor"],
            "counterfactuals": [
                {
                    "intervention": f"remove or reverse the decisive factor `{factor}`",
                    "expected_effect": "The answer should be re-evaluated because the causal basis changes.",
                    "answer": "",
                    "question": "",
                }
            ],
            "answer": sample.gold_answer,
        }

    def fill_knowledge_sample(self, sample: MedSample, reasoning_prefix: str) -> str:
        facts = split_sentences(sample.knowledge)
        if facts:
            return facts[0]
        return sample.gold_answer
