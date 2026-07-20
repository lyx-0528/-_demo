from __future__ import annotations

from dataclasses import asdict
import inspect
import json
from statistics import mean
from typing import Any, Optional

from ..knowledge_tokens import register_knowledge_special_tokens
from .config import TrainingRuntimeConfig
from .dataset import IGNORE_INDEX, build_tokenized_supervision_dataset
from .runtime import attach_lora, resolve_torch_dtype
from .trainer import SupervisionDataCollator, build_weighted_trainer_class


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"answer": 0, "counterfactual": 0, "policy": 0}
    sample_weights: list[float] = []
    frontdoor_enabled = 0
    frontdoor_gains: list[float] = []
    frontdoor_ttl_losses: list[float] = []
    for record in records:
        task_type = str(record.get("task_type", "")).strip()
        if task_type in counts:
            counts[task_type] += 1
        sample_weights.append(float(record.get("sample_weight", 1.0) or 1.0))
        if bool(record.get("frontdoor_enabled", False)):
            frontdoor_enabled += 1
            frontdoor_gains.append(float(record.get("frontdoor_weighted_gain", 0.0) or 0.0))
            frontdoor_ttl_losses.append(float(record.get("frontdoor_ttl_loss", 0.0) or 0.0))

    summary = {
        "counts": counts,
        "avg_sample_weight": mean(sample_weights) if sample_weights else 0.0,
        "max_sample_weight": max(sample_weights) if sample_weights else 0.0,
        "frontdoor_enabled_records": frontdoor_enabled,
    }
    if frontdoor_gains:
        summary["avg_frontdoor_weighted_gain"] = mean(frontdoor_gains)
        summary["avg_frontdoor_ttl_loss"] = mean(frontdoor_ttl_losses)
    return summary


def train_on_supervision_records(
    train_records: list[dict[str, Any]],
    *,
    config: TrainingRuntimeConfig,
    eval_records: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

    if not train_records:
        raise ValueError("No supervision records were loaded for training.")

    config.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        torch_dtype=resolve_torch_dtype(config),
    )
    register_knowledge_special_tokens(tokenizer, model)
    model = attach_lora(model, config)

    if config.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    train_dataset = build_tokenized_supervision_dataset(train_records, tokenizer, config.max_length)
    eval_dataset = (
        build_tokenized_supervision_dataset(eval_records or [], tokenizer, config.max_length)
        if eval_records
        else None
    )

    pad_to_multiple = 8 if torch.cuda.is_available() else None
    data_collator = SupervisionDataCollator(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=IGNORE_INDEX,
        pad_to_multiple_of=pad_to_multiple,
    )

    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps if config.eval_steps > 0 else None,
        eval_strategy="steps" if eval_dataset is not None and config.eval_steps > 0 else "no",
        save_strategy="steps",
        bf16=config.bf16 and torch.cuda.is_available(),
        fp16=config.fp16 and torch.cuda.is_available(),
        report_to=[] if config.report_to == "none" else [config.report_to],
        remove_unused_columns=False,
        seed=config.seed,
    )

    trainer_cls = build_weighted_trainer_class()
    trainer_kwargs = {
        "runtime_config": config,
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }
    trainer_signature = inspect.signature(trainer_cls.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = trainer_cls(
        **trainer_kwargs,
    )
    train_result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)

    summary = {
        "config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
        },
        "num_train_records": len(train_dataset),
        "num_eval_records": len(eval_dataset) if eval_dataset is not None else 0,
        "train_record_stats": _summarize_records(train_records),
        "train_metrics": train_result.metrics,
        "loss_weights": {
            "answer": config.answer_loss_weight,
            "counterfactual": config.counterfactual_loss_weight,
            "policy": config.policy_loss_weight,
            "use_sample_weight": config.use_sample_weight,
            "normalize_weighted_loss": config.normalize_weighted_loss,
        },
    }
    if eval_records is not None:
        summary["eval_record_stats"] = _summarize_records(eval_records)
    if eval_dataset is not None:
        summary["eval_metrics"] = trainer.evaluate()

    (config.output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
