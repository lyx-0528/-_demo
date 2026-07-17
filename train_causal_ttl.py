from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal_ttl.hf_training import TrainingRuntimeConfig, load_supervision_records, train_on_supervision_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a local causal TTL model on exported supervision records.")
    parser.add_argument("--train-records", type=Path, required=True, help="Path to supervision_dataset.jsonl.")
    parser.add_argument("--eval-records", type=Path, default=None, help="Optional path to eval supervision jsonl.")
    parser.add_argument("--task-types", nargs="*", default=None, help="Optional subset of task types to train on.")
    parser.add_argument("--model-name-or-path", required=True, help="Base causal LM model path or HF id.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to save the trained model or adapter.")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--lr-scheduler-type", default="constant")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--finetuning-type", choices=("lora", "full"), default="lora")
    parser.add_argument("--lora-target", default="q_proj,v_proj")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--answer-loss-weight", type=float, default=1.0)
    parser.add_argument("--counterfactual-loss-weight", type=float, default=1.0)
    parser.add_argument("--policy-loss-weight", type=float, default=1.0)
    parser.add_argument("--no-sample-weight", action="store_true")
    parser.add_argument("--no-normalize-weighted-loss", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_records = load_supervision_records(args.train_records, task_types=args.task_types)
    eval_records = (
        load_supervision_records(args.eval_records, task_types=args.task_types)
        if args.eval_records is not None
        else None
    )

    config = TrainingRuntimeConfig(
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir.resolve(),
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        bf16=not args.no_bf16,
        fp16=args.fp16,
        seed=args.seed,
        finetuning_type=args.finetuning_type,
        lora_target=args.lora_target,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        answer_loss_weight=args.answer_loss_weight,
        counterfactual_loss_weight=args.counterfactual_loss_weight,
        policy_loss_weight=args.policy_loss_weight,
        use_sample_weight=not args.no_sample_weight,
        normalize_weighted_loss=not args.no_normalize_weighted_loss,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )
    summary = train_on_supervision_records(train_records, config=config, eval_records=eval_records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
