from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal_ttl import build_teacher, load_samples_auto
from causal_ttl.hf_inference import InferenceRuntimeConfig, predict_samples, write_predictions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local causal TTL inference.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to a prepared test.json or raw CSV/JSON dataset.")
    parser.add_argument("--model-name-or-path", required=True, help="Base model path or HF id.")
    parser.add_argument("--predictions-path", type=Path, required=True, help="Where to write prediction JSONL.")
    parser.add_argument("--adapter-path", type=Path, default=None, help="Optional LoRA adapter directory.")
    parser.add_argument("--mode", choices=("fact", "v6", "v7"), default="v7")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--knowledge-at-inference", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--max-cloud-calls", type=int, default=3)
    parser.add_argument("--cloud-budget-ratio", type=float, default=1.0)
    parser.add_argument("--teacher-backend", choices=("heuristic", "api"), default="heuristic")
    parser.add_argument("--teacher-api-base", default=None)
    parser.add_argument("--teacher-api-key", default=None)
    parser.add_argument("--teacher-model", default=None)
    parser.add_argument("--teacher-temperature", type=float, default=0.0)
    parser.add_argument("--teacher-max-new-tokens", type=int, default=512)
    parser.add_argument("--teacher-cache-dir", type=Path, default=Path("outputs") / "teacher_cache")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    samples = load_samples_auto(args.dataset.resolve(), limit=args.limit)

    teacher = build_teacher(
        args.teacher_backend,
        api_base=args.teacher_api_base,
        api_key=args.teacher_api_key,
        model=args.teacher_model,
        temperature=args.teacher_temperature,
        max_new_tokens=args.teacher_max_new_tokens,
        cache_dir=str(args.teacher_cache_dir),
    )
    config = InferenceRuntimeConfig(
        model_name_or_path=args.model_name_or_path,
        predictions_path=args.predictions_path.resolve(),
        adapter_path=args.adapter_path.resolve() if args.adapter_path is not None else None,
        mode=args.mode,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        top_p=args.top_p,
        knowledge_at_inference=args.knowledge_at_inference,
        chunk_size=args.chunk_size,
        max_cloud_calls=args.max_cloud_calls,
        cloud_budget_ratio=args.cloud_budget_ratio,
    )
    rows = predict_samples(samples, config=config, teacher=teacher)
    summary = write_predictions(rows, config.predictions_path, config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
