from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean

from causal_ttl import CausalTTLPipeline, PipelineConfig, load_samples_auto


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = PROJECT_ROOT.parent / "MedThink" / "PrecisionBoost" / "q&a.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the causal TTL reference pipeline.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to a CSV or JSON dataset. Defaults to the MedThink q&a.csv file.",
    )
    parser.add_argument("--limit", type=int, default=5, help="How many samples to run.")
    parser.add_argument("--num-paths", type=int, default=8, help="How many reasoning paths to sample.")
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.45,
        help="Similarity threshold used when clustering reasoning traces.",
    )
    parser.add_argument("--max-clusters", type=int, default=4, help="Maximum number of clusters to keep.")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for a JSON report.",
    )
    return parser


def summarize_results(results) -> dict:
    baseline_scores = [result.baseline_score for result in results]
    final_scores = [result.final_score for result in results]
    improved = sum(1 for result in results if result.final_score > result.baseline_score)
    return {
        "num_samples": len(results),
        "avg_baseline_score": mean(baseline_scores) if baseline_scores else 0.0,
        "avg_final_score": mean(final_scores) if final_scores else 0.0,
        "improved_samples": improved,
    }


def print_result(result) -> None:
    print(f"[{result.sample.sample_id}] {result.sample.question}")
    print(f"  gold      : {result.sample.gold_answer}")
    print(f"  baseline  : {result.baseline_answer} (score={result.baseline_score:.3f})")
    print(f"  final     : {result.final_answer} (score={result.final_score:.3f})")
    print(f"  chosen    : {result.chosen_cluster_id or 'none'}")
    if result.interventions:
        best = result.interventions[0]
        print(
            "  diagnosis : "
            f"{best.diagnosis.error_type} | gain={best.causal_gain:.3f} | rule={best.diagnosis.reasoning_rule}"
        )
    print()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    samples = load_samples_auto(args.dataset, limit=args.limit)
    config = PipelineConfig(
        num_reasoning_paths=args.num_paths,
        cluster_similarity_threshold=args.cluster_threshold,
        max_clusters=args.max_clusters,
    )
    pipeline = CausalTTLPipeline(config=config)
    results = pipeline.run_dataset(samples)

    for result in results:
        print_result(result)

    summary = summarize_results(results)
    print("Summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(config),
            "summary": summary,
            "results": [asdict(result) for result in results],
        }
        args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON report saved to: {args.json_out}")


if __name__ == "__main__":
    main()
