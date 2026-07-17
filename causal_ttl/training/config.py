from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TrainingRuntimeConfig:
    model_name_or_path: str
    output_dir: Path
    max_length: int = 2048
    learning_rate: float = 5.0e-5
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.0
    lr_scheduler_type: str = "constant"
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 0
    bf16: bool = True
    fp16: bool = False
    seed: int = 17
    finetuning_type: str = "lora"
    lora_target: str = "q_proj,v_proj"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    trust_remote_code: bool = True
    gradient_checkpointing: bool = True
    report_to: str = "none"
    answer_loss_weight: float = 1.0
    counterfactual_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    use_sample_weight: bool = True
    normalize_weighted_loss: bool = True
