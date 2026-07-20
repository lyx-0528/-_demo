from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal_ttl.evaluation import evaluate_predictions_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate causal TTL prediction JSONL.")
    parser.add_argument("--predictions", type=Path, required=True, help="Path to the prediction JSONL file.")
    parser.add_argument(
        "--metric",
        choices=("auto", "text", "gsm8k"),
        default="auto",
        help="Evaluation metric. Use `gsm8k` for numeric exact match on arithmetic datasets.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = evaluate_predictions_file(args.predictions.resolve(), metric=args.metric)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
