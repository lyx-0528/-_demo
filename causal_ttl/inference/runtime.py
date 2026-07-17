from __future__ import annotations

import hashlib

from .config import InferenceRuntimeConfig


def resolve_torch_dtype():
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.bfloat16


def allow_cloud_call(cloud_budget_ratio: float, sample_key: str) -> bool:
    cloud_budget_ratio = max(0.0, min(1.0, float(cloud_budget_ratio)))
    if cloud_budget_ratio >= 1.0:
        return True
    if cloud_budget_ratio <= 0.0:
        return False
    bucket = int(hashlib.sha256(sample_key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return bucket < cloud_budget_ratio


def load_model_and_tokenizer(config: InferenceRuntimeConfig):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.adapter_path or config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=resolve_torch_dtype(),
    )
    if config.adapter_path is not None:
        try:
            from peft import PeftModel
        except ModuleNotFoundError as exc:
            raise RuntimeError("PEFT is required to load a LoRA adapter. Install `peft`.") from exc
        model = PeftModel.from_pretrained(model, str(config.adapter_path))

    model.eval()
    return model, tokenizer
