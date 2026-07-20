from __future__ import annotations

import hashlib

from .config import InferenceRuntimeConfig
from ..knowledge_tokens import register_knowledge_special_tokens


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
    import torch
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
        device_map="auto" if torch.cuda.is_available() else None,
    )
    register_knowledge_special_tokens(tokenizer, model)
    current_vocab_size = model.get_input_embeddings().weight.shape[0]
    target_vocab_size = len(tokenizer)
    if current_vocab_size != target_vocab_size:
        model.resize_token_embeddings(target_vocab_size)
    if config.adapter_path is not None:
        try:
            from peft import PeftModel
        except ModuleNotFoundError as exc:
            raise RuntimeError("PEFT is required to load a LoRA adapter. Install `peft`.") from exc
        model = PeftModel.from_pretrained(model, str(config.adapter_path))

    if not hasattr(model, "hf_device_map") and torch.cuda.is_available():
        model = model.to("cuda")

    model.eval()
    return model, tokenizer
