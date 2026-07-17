from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal_ttl.evaluation import evaluate_predictions_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate causal TTL prediction JSONL.")
    parser.add_argument("--predictions", type=Path, required=True, help="Path to the prediction JSONL file.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = evaluate_predictions_file(args.predictions.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
