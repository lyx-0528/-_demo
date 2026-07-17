from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from causal_ttl import FrontDoorConfig, build_teacher, config_from_mode, export_supervision_dataset, load_samples_auto
from causal_ttl.evaluation import evaluate_predictions_file
from causal_ttl.hf_inference import InferenceRuntimeConfig, predict_samples, write_predictions
from causal_ttl.hf_training import TrainingRuntimeConfig, load_supervision_records, train_on_supervision_records


PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local causal TTL workflow from a config file.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a JSON or YAML config.")
    return parser


def load_config(path: Path) -> dict:
    raw_text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"Config {path} is not valid JSON, and PyYAML is not available. "
                "Install PyYAML or rewrite the config in JSON syntax."
            ) from exc
        payload = yaml.safe_load(raw_text)
    if not isinstance(payload, dict):
        raise ValueError(f"Config {path} did not parse into a dictionary.")
    return payload


def resolve_path(path_text: str | None) -> Path | None:
    if path_text is None:
        return None
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def main() -> None:
    args = build_parser().parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)

    experiment_cfg = config.get("experiment", {})
    data_cfg = config.get("data", {})
    teacher_cfg = config.get("teacher", {})
    supervision_cfg = config.get("supervision", {})
    frontdoor_cfg = config.get("frontdoor", {})
    training_cfg = config.get("training", {})
    inference_cfg = config.get("inference", {})

    output_root = resolve_path(experiment_cfg.get("output_dir", "outputs/causal_workflow"))
    assert output_root is not None
    output_root.mkdir(parents=True, exist_ok=True)

    train_samples = load_samples_auto(resolve_path(data_cfg["train_dataset"]), limit=data_cfg.get("limit"))
    eval_samples = load_samples_auto(resolve_path(data_cfg["eval_dataset"]), limit=data_cfg.get("eval_limit"))

    teacher = build_teacher(
        teacher_cfg.get("backend", "heuristic"),
        api_base=teacher_cfg.get("api_base"),
        api_key=teacher_cfg.get("api_key"),
        model=teacher_cfg.get("model"),
        temperature=teacher_cfg.get("temperature", 0.0),
        max_new_tokens=teacher_cfg.get("max_new_tokens", 512),
        cache_dir=str(resolve_path(teacher_cfg.get("cache_dir", str(output_root / "teacher_cache")))),
    )

    supervision_mode = supervision_cfg.get("mode", "v7")
    frontdoor = FrontDoorConfig(
        enabled=frontdoor_cfg.get("enabled", False),
        num_reasoning_paths=frontdoor_cfg.get("num_reasoning_paths", 8),
        cluster_similarity_threshold=frontdoor_cfg.get("cluster_similarity_threshold", 0.45),
        max_clusters=frontdoor_cfg.get("max_clusters", 4),
        causal_margin=frontdoor_cfg.get("causal_margin", 0.15),
        min_feedback_gain=frontdoor_cfg.get("min_feedback_gain", 0.05),
        answer_weight_floor=frontdoor_cfg.get("answer_weight_floor", 1.0),
        answer_weight_scale=frontdoor_cfg.get("answer_weight_scale", 1.0),
        counterfactual_weight_scale=frontdoor_cfg.get("counterfactual_weight_scale", 1.0),
        policy_weight_scale=frontdoor_cfg.get("policy_weight_scale", 1.0),
        max_sample_weight=frontdoor_cfg.get("max_sample_weight", 4.0),
        seed=frontdoor_cfg.get("seed", experiment_cfg.get("seed", 17)),
    )
    supervision_export_config = config_from_mode(
        supervision_mode,
        cloud_budget_ratio=supervision_cfg.get("cloud_budget_ratio", 1.0),
        frontdoor=frontdoor,
    )
    supervision_dir = output_root / "supervision_train"
    train_supervision_summary = export_supervision_dataset(train_samples, teacher, supervision_dir, supervision_export_config)

    eval_supervision_dir = output_root / "supervision_eval"
    eval_supervision_summary = export_supervision_dataset(eval_samples, teacher, eval_supervision_dir, supervision_export_config)

    train_records = load_supervision_records(
        supervision_dir / "supervision_dataset.jsonl",
        task_types=training_cfg.get("task_types"),
    )
    eval_records = load_supervision_records(
        eval_supervision_dir / "supervision_dataset.jsonl",
        task_types=training_cfg.get("task_types"),
    )

    model_dir = output_root / "model"
    training_summary = train_on_supervision_records(
        train_records,
        eval_records=eval_records if training_cfg.get("use_eval_records", False) else None,
        config=TrainingRuntimeConfig(
            model_name_or_path=training_cfg["model_name_or_path"],
            output_dir=model_dir,
            max_length=training_cfg.get("max_length", 2048),
            learning_rate=training_cfg.get("learning_rate", 5.0e-5),
            num_train_epochs=training_cfg.get("num_train_epochs", 1.0),
            per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 1),
            per_device_eval_batch_size=training_cfg.get("per_device_eval_batch_size", 1),
            gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),
            warmup_ratio=training_cfg.get("warmup_ratio", 0.0),
            lr_scheduler_type=training_cfg.get("lr_scheduler_type", "constant"),
            logging_steps=training_cfg.get("logging_steps", 10),
            save_steps=training_cfg.get("save_steps", 200),
            eval_steps=training_cfg.get("eval_steps", 0),
            bf16=training_cfg.get("bf16", True),
            fp16=training_cfg.get("fp16", False),
            seed=experiment_cfg.get("seed", 17),
            finetuning_type=training_cfg.get("finetuning_type", "lora"),
            lora_target=training_cfg.get("lora_target", "q_proj,v_proj"),
            lora_rank=training_cfg.get("lora_rank", 8),
            lora_alpha=training_cfg.get("lora_alpha", 16),
            lora_dropout=training_cfg.get("lora_dropout", 0.05),
            answer_loss_weight=training_cfg.get("answer_loss_weight", 1.0),
            counterfactual_loss_weight=training_cfg.get("counterfactual_loss_weight", 1.0),
            policy_loss_weight=training_cfg.get("policy_loss_weight", 1.0),
            use_sample_weight=training_cfg.get("use_sample_weight", True),
            normalize_weighted_loss=training_cfg.get("normalize_weighted_loss", True),
            gradient_checkpointing=training_cfg.get("gradient_checkpointing", True),
        ),
    )

    predictions_path = output_root / "predictions.jsonl"
    inference_runtime = InferenceRuntimeConfig(
        model_name_or_path=inference_cfg.get("base_model_name_or_path", training_cfg["model_name_or_path"]),
        predictions_path=predictions_path,
        adapter_path=model_dir,
        mode=inference_cfg.get("mode", supervision_mode),
        max_length=inference_cfg.get("max_length", training_cfg.get("max_length", 2048)),
        max_new_tokens=inference_cfg.get("max_new_tokens", 512),
        temperature=inference_cfg.get("temperature", 0.0),
        do_sample=inference_cfg.get("do_sample", False),
        top_p=inference_cfg.get("top_p", 1.0),
        knowledge_at_inference=inference_cfg.get("knowledge_at_inference", supervision_mode in {"fact", "v6"}),
        chunk_size=inference_cfg.get("chunk_size", 64),
        max_cloud_calls=inference_cfg.get("max_cloud_calls", 3),
        cloud_budget_ratio=inference_cfg.get("cloud_budget_ratio", supervision_cfg.get("cloud_budget_ratio", 1.0)),
    )
    prediction_rows = predict_samples(eval_samples, config=inference_runtime, teacher=teacher)
    prediction_summary = write_predictions(prediction_rows, predictions_path, inference_runtime)
    evaluation_summary = evaluate_predictions_file(predictions_path)

    workflow_summary = {
        "experiment": {
            "name": experiment_cfg.get("name", config_path.stem),
            "config_path": str(config_path),
            "output_dir": str(output_root),
            "seed": experiment_cfg.get("seed", 17),
        },
        "supervision": {
            "mode": supervision_mode,
            "config": asdict(supervision_export_config),
            "train_export": train_supervision_summary,
            "eval_export": eval_supervision_summary,
        },
        "frontdoor": asdict(frontdoor),
        "training": training_summary,
        "prediction": prediction_summary,
        "evaluation": evaluation_summary,
    }
    (output_root / "workflow_summary.json").write_text(
        json.dumps(workflow_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(workflow_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
