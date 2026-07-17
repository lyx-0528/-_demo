from __future__ import annotations

import re

from ..knowledge_tokens import (
    KNOWLEDGE_TOKEN,
    has_unfilled_knowledge,
    replace_first_unfilled_knowledge,
    wrap_filled_knowledge,
)
from ..schema import MedSample
from ..teachers import BaseTeacher
from .config import InferenceRuntimeConfig
from .runtime import allow_cloud_call


_ANSWER_RE = re.compile(r"(?is)answer\s*:\s*(.*)\Z")
_ASK_CLOUD_RE = re.compile(
    r"Action:\s*ASK_CLOUD\s*\nQuery:\s*(?P<query>.+?)(?=\n(?:Observation:|Causal:|Need:|Step\s+\d+\s*\||Answer:)|\Z)",
    flags=re.DOTALL,
)
_STEP_CONTEXT_RE = re.compile(
    r"(?:^|\n)(?P<step>Step\s+\d+\s*\|\s*Goal:\s*(?P<goal>[^\n]*).*?)(?=\nStep\s+\d+\s*\||\nAnswer:|\Z)",
    flags=re.DOTALL,
)
_UNCERTAINTY_RE = re.compile(
    r"\b("
    r"i do not know|i don't know|cannot determine|can't determine|need more information|"
    r"insufficient information|not enough information|"
    r"需要更多信息|无法确定|不能确定|不知道"
    r")\b",
    flags=re.IGNORECASE,
)


def extract_final_answer(text: str) -> str:
    match = _ANSWER_RE.search(text or "")
    answer = match.group(1).strip() if match else (text or "").strip()
    answer = re.sub(r"</?knowledge>", "", answer).replace("<|im_end|>", "").strip()
    return answer


def find_unanswered_cloud_query(text: str) -> str:
    for match in reversed(list(_ASK_CLOUD_RE.finditer(text))):
        tail = text[match.end() :]
        next_step = re.search(r"\n(?:Step\s+\d+\s*\||Answer:)", tail)
        block_tail = tail[: next_step.start()] if next_step else tail
        if "Observation:" not in block_tail:
            return re.sub(r"\s+", " ", match.group("query")).strip()
    return ""


def find_need_knowledge_query(text: str) -> str:
    if not has_unfilled_knowledge(text):
        return ""

    for match in reversed(list(_STEP_CONTEXT_RE.finditer(text))):
        block = match.group("step")
        if KNOWLEDGE_TOKEN not in block:
            continue

        explicit_query = re.search(
            r"Action:\s*ASK_CLOUD\s*\nQuery:\s*(.*?)(?=\n(?:Observation:|Causal:|Need:)|\Z)",
            block,
            flags=re.DOTALL,
        )
        if explicit_query:
            query = re.sub(r"\s+", " ", explicit_query.group(1)).strip()
            if query:
                return query

        goal = re.sub(r"\s+", " ", match.group("goal")).strip()
        causal_match = re.search(r"Causal:\s*(.*?)(?=\n(?:Need:|Action:|Query:|Observation:)|\Z)", block, re.DOTALL)
        causal = re.sub(r"\s+", " ", causal_match.group(1)).strip() if causal_match else ""
        if causal:
            return f"What external knowledge is needed for this causal step? Goal: {goal}. Causal context: {causal}"
        return f"What external knowledge is needed for this causal step? Goal: {goal}"

    return "Fill the next missing domain fact required by the current causal reasoning."


def should_ask_for_uncertainty(text: str, previous_text: str) -> bool:
    if len(text) <= len(previous_text):
        return False
    return _UNCERTAINTY_RE.search(text[len(previous_text) :]) is not None


def generate_direct(
    model,
    tokenizer,
    prompt_text: str,
    *,
    max_new_tokens: int,
    temperature: float,
    do_sample: bool,
    top_p: float,
) -> str:
    import torch

    device = next(model.parameters()).device
    encoded = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    generate_kwargs: dict[str, object] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature if do_sample else None,
        "top_p": top_p if do_sample else None,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "attention_mask": attention_mask,
    }
    generate_kwargs = {key: value for key, value in generate_kwargs.items() if value is not None}

    with torch.no_grad():
        output_ids = model.generate(input_ids=input_ids, **generate_kwargs)
    return tokenizer.decode(output_ids[0, input_ids.size(1) :], skip_special_tokens=False)


def interactive_generate(
    model,
    tokenizer,
    sample: MedSample,
    prompt_text: str,
    *,
    teacher: BaseTeacher,
    config: InferenceRuntimeConfig,
) -> tuple[str, list[dict[str, str]]]:
    import torch

    device = next(model.parameters()).device
    generated = ""
    interactions: list[dict[str, str]] = []
    tokens_used = 0
    rounds = 0

    while tokens_used < config.max_new_tokens and rounds <= config.max_cloud_calls:
        previous = generated
        encoded = tokenizer(
            prompt_text + generated,
            return_tensors="pt",
            truncation=True,
            max_length=config.max_length,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        remaining = min(config.chunk_size, config.max_new_tokens - tokens_used)

        generate_kwargs: dict[str, object] = {
            "max_new_tokens": remaining,
            "do_sample": config.do_sample,
            "temperature": config.temperature if config.do_sample else None,
            "top_p": config.top_p if config.do_sample else None,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "attention_mask": attention_mask,
        }
        generate_kwargs = {key: value for key, value in generate_kwargs.items() if value is not None}

        with torch.no_grad():
            output_ids = model.generate(input_ids=input_ids, **generate_kwargs)

        new_ids = output_ids[0, input_ids.size(1) :]
        if new_ids.numel() == 0:
            break

        generated += tokenizer.decode(new_ids, skip_special_tokens=False)
        tokens_used += int(new_ids.numel())

        query = find_unanswered_cloud_query(generated)
        trigger = "ASK_CLOUD" if query else ""
        if not query and has_unfilled_knowledge(generated):
            query = find_need_knowledge_query(generated)
            trigger = KNOWLEDGE_TOKEN
        if not query and should_ask_for_uncertainty(generated, previous):
            query = (
                "The local model became uncertain while answering. "
                "Provide the minimal factual or causal knowledge needed to continue."
            )
            trigger = "UNCERTAIN"

        if query and rounds < config.max_cloud_calls:
            rounds += 1
            if not allow_cloud_call(config.cloud_budget_ratio, sample.sample_id):
                fact = "Cloud query skipped by budget; continue with local causal reasoning."
                interactions.append(
                    {
                        "trigger": trigger,
                        "query": query,
                        "knowledge": fact,
                        "cloud_called": "false",
                        "skip_reason": "budget",
                    }
                )
            else:
                fact = teacher.fill_knowledge_sample(sample, f"{generated}\n\nQUERY:\n{query}")
                interactions.append(
                    {
                        "trigger": trigger,
                        "query": query,
                        "knowledge": fact,
                        "cloud_called": "true",
                    }
                )

            if trigger == KNOWLEDGE_TOKEN:
                generated = replace_first_unfilled_knowledge(generated, fact)
            else:
                generated = f"{generated.rstrip()}\nObservation: {wrap_filled_knowledge(fact)}\n"
            continue

        if "Answer:" in generated and not has_unfilled_knowledge(generated):
            break

    return generated, interactions
