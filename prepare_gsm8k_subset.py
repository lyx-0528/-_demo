from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from causal_ttl import load_samples_auto


DEFAULT_SOURCE = Path(__file__).resolve().parent.parent / "CloudDeviceTTL" / "data" / "AdaptEval" / "gsm8k_random_5k.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "processed" / "gsm8k_500"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a GSM8K subset for causal TTL experiments.")
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE, help="Path to gsm8k_random_5k.json.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to write train/test json files.")
    parser.add_argument("--limit", type=int, default=500, help="Total number of samples to keep.")
    parser.add_argument("--test-size", type=int, default=500, help="Number of evaluation samples.")
    parser.add_argument("--train-size", type=int, default=500, help="Number of training samples.")
    parser.add_argument("--seed", type=int, default=17, help="Random seed.")
    return parser


def sample_to_record(sample, split: str) -> dict:
    return {
        "id": sample.sample_id,
        "question": sample.question,
        "answer": sample.gold_answer,
        "knowledge": sample.knowledge,
        "source": sample.source,
        "split": split,
    }


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    samples = load_samples_auto(args.input.resolve())
    if not samples:
        raise SystemExit(f"No samples loaded from {args.input}")

    shuffled = list(samples)
    random.Random(args.seed).shuffle(shuffled)

    required = max(args.train_size, 0) + max(args.test_size, 0)
    if required > len(shuffled):
        raise SystemExit(f"Requested {required} samples but only found {len(shuffled)}.")

    train_samples = shuffled[: args.train_size]
    test_samples = shuffled[args.train_size : args.train_size + args.test_size]

    output_dir = args.output_dir.resolve()
    train_records = [sample_to_record(sample, "train") for sample in train_samples]
    test_records = [sample_to_record(sample, "test") for sample in test_samples]

    write_json(output_dir / "train.json", train_records)
    write_json(output_dir / "valid.json", [])
    write_json(output_dir / "test.json", test_records)
    write_json(
        output_dir / "manifest.json",
        {
            "seed": args.seed,
            "source": str(args.input.resolve()),
            "train_size": len(train_records),
            "test_size": len(test_records),
            "valid_size": 0,
        },
    )

    print(f"Prepared GSM8K subset under: {output_dir}")
    print(json.dumps({"train_size": len(train_records), "test_size": len(test_records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
