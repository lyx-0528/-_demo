from __future__ import annotations

from .config import TrainingRuntimeConfig


def resolve_torch_dtype(config: TrainingRuntimeConfig):
    import torch

    if not torch.cuda.is_available():
        return None
    if config.bf16:
        return torch.bfloat16
    if config.fp16:
        return torch.float16
    return None


def attach_lora(model, config: TrainingRuntimeConfig):
    if config.finetuning_type.strip().lower() != "lora":
        return model

    try:
        from peft import LoraConfig, get_peft_model
    except ModuleNotFoundError as exc:
        raise RuntimeError("PEFT is required for LoRA fine-tuning. Install `peft`.") from exc

    target_modules = [module.strip() for module in config.lora_target.split(",") if module.strip()]
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)
