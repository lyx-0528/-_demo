from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from causal_ttl import load_samples_auto


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize local datasets for later cloud runs.")
    parser.add_argument("--input", type=Path, default=None, help="Single CSV/JSON dataset to split.")
    parser.add_argument("--train-input", type=Path, default=None, help="Optional explicit train dataset.")
    parser.add_argument("--test-input", type=Path, default=None, help="Optional explicit test dataset.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
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


def write_split(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def split_samples(samples, train_ratio: float, valid_ratio: float, seed: int):
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * train_ratio)
    valid_end = train_end + int(total * valid_ratio)
    return shuffled[:train_end], shuffled[train_end:valid_end], shuffled[valid_end:]


def main() -> None:
    args = build_parser().parse_args()

    if args.input is None and (args.train_input is None or args.test_input is None):
        raise SystemExit("Provide either --input or both --train-input and --test-input.")

    if args.input is not None:
        samples = load_samples_auto(args.input, limit=args.limit)
        train_samples, valid_samples, test_samples = split_samples(
            samples,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
        )
    else:
        train_samples = load_samples_auto(args.train_input, limit=args.limit)
        test_samples = load_samples_auto(args.test_input, limit=args.limit)
        valid_samples = []

    output_dir = args.output_dir.resolve()
    train_records = [sample_to_record(sample, "train") for sample in train_samples]
    valid_records = [sample_to_record(sample, "valid") for sample in valid_samples]
    test_records = [sample_to_record(sample, "test") for sample in test_samples]

    write_split(output_dir / "train.json", train_records)
    write_split(output_dir / "valid.json", valid_records)
    write_split(output_dir / "test.json", test_records)
    manifest = {
        "seed": args.seed,
        "train_size": len(train_records),
        "valid_size": len(valid_records),
        "test_size": len(test_records),
        "input": str(args.input) if args.input is not None else None,
        "train_input": str(args.train_input) if args.train_input is not None else None,
        "test_input": str(args.test_input) if args.test_input is not None else None,
    }
    write_split(output_dir / "manifest.json", manifest)

    print(f"Prepared dataset under: {output_dir}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
