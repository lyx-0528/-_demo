from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal_ttl import FrontDoorConfig, build_teacher, config_from_mode, export_supervision_dataset, load_samples_auto


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_ROOT.parent / "MedThink" / "PrecisionBoost" / "q&a.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export CloudDeviceTTL-style supervision records from the local causal TTL project."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to a CSV or JSON dataset.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where JSONL artifacts will be written.")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit.")
    parser.add_argument(
        "--mode",
        choices=("fact", "v6", "v7"),
        default="v7",
        help="Export preset. `fact` mirrors causal_collaborative_ttl_fact, `v6` mirrors v6 loss samples, `v7` mirrors structured tool-use supervision.",
    )
    parser.add_argument(
        "--teacher-backend",
        choices=("heuristic", "api"),
        default="heuristic",
        help="Teacher backend. Use `heuristic` for local dry runs or `api` for a real teacher.",
    )
    parser.add_argument("--teacher-api-base", default=None, help="Override TEACHER_API_BASE for the API backend.")
    parser.add_argument("--teacher-api-key", default=None, help="Override TEACHER_API_KEY for the API backend.")
    parser.add_argument("--teacher-model", default=None, help="Override TEACHER_MODEL for the API backend.")
    parser.add_argument("--teacher-temperature", type=float, default=0.0, help="Teacher temperature.")
    parser.add_argument("--teacher-max-new-tokens", type=int, default=512, help="Teacher max_new_tokens.")
    parser.add_argument(
        "--teacher-cache-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "teacher_cache",
        help="Directory used to cache teacher responses.",
    )
    parser.add_argument(
        "--cloud-budget-ratio",
        type=float,
        default=1.0,
        help="Fraction of fact-bearing samples that will be taught to issue ASK_CLOUD steps.",
    )
    parser.add_argument("--enable-frontdoor", action="store_true", help="Run the PPT front-door pipeline and export its signals.")
    parser.add_argument("--frontdoor-num-paths", type=int, default=8)
    parser.add_argument("--frontdoor-similarity-threshold", type=float, default=0.45)
    parser.add_argument("--frontdoor-max-clusters", type=int, default=4)
    parser.add_argument("--frontdoor-causal-margin", type=float, default=0.15)
    parser.add_argument("--frontdoor-min-feedback-gain", type=float, default=0.05)
    parser.add_argument("--frontdoor-answer-weight-floor", type=float, default=1.0)
    parser.add_argument("--frontdoor-answer-weight-scale", type=float, default=1.0)
    parser.add_argument("--frontdoor-counterfactual-weight-scale", type=float, default=1.0)
    parser.add_argument("--frontdoor-policy-weight-scale", type=float, default=1.0)
    parser.add_argument("--frontdoor-max-sample-weight", type=float, default=4.0)
    parser.add_argument("--frontdoor-seed", type=int, default=17)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    samples = load_samples_auto(args.dataset, limit=args.limit)
    teacher = build_teacher(
        args.teacher_backend,
        api_base=args.teacher_api_base,
        api_key=args.teacher_api_key,
        model=args.teacher_model,
        temperature=args.teacher_temperature,
        max_new_tokens=args.teacher_max_new_tokens,
        cache_dir=str(args.teacher_cache_dir),
    )
    frontdoor = FrontDoorConfig(
        enabled=args.enable_frontdoor,
        num_reasoning_paths=args.frontdoor_num_paths,
        cluster_similarity_threshold=args.frontdoor_similarity_threshold,
        max_clusters=args.frontdoor_max_clusters,
        causal_margin=args.frontdoor_causal_margin,
        min_feedback_gain=args.frontdoor_min_feedback_gain,
        answer_weight_floor=args.frontdoor_answer_weight_floor,
        answer_weight_scale=args.frontdoor_answer_weight_scale,
        counterfactual_weight_scale=args.frontdoor_counterfactual_weight_scale,
        policy_weight_scale=args.frontdoor_policy_weight_scale,
        max_sample_weight=args.frontdoor_max_sample_weight,
        seed=args.frontdoor_seed,
    )
    config = config_from_mode(args.mode, cloud_budget_ratio=args.cloud_budget_ratio, frontdoor=frontdoor)
    summary = export_supervision_dataset(samples, teacher, args.output_dir.resolve(), config)

    report = {
        "dataset": str(args.dataset.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "teacher_backend": args.teacher_backend,
        "mode": args.mode,
        "summary": summary,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
