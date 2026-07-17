from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from baselines import build_baseline
from causal_ttl import PipelineConfig, load_samples_auto
from metrics import serialize_results, summarize_results, write_csv_report, write_json_report


PROJECT_ROOT = Path(__file__).resolve().parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local dry run or a cloud-ready experiment.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a YAML experiment config.")
    parser.add_argument("--limit", type=int, default=None, help="Optional runtime override for sample count.")
    return parser


def load_config(path: Path) -> dict:
    raw_text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as json_error:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as yaml_error:
            raise RuntimeError(
                f"Config {path} is not valid JSON, and PyYAML is not available. "
                "Either install PyYAML or rewrite the config file in JSON syntax."
            ) from yaml_error
        payload = yaml.safe_load(raw_text)
    if not isinstance(payload, dict):
        raise ValueError(f"Config {path} did not parse into a dictionary.")
    return payload


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def pipeline_config_from_dict(payload: dict) -> PipelineConfig:
    return PipelineConfig(
        num_reasoning_paths=payload.get("num_reasoning_paths", 8),
        cluster_similarity_threshold=payload.get("cluster_similarity_threshold", 0.45),
        max_clusters=payload.get("max_clusters", 4),
        causal_margin=payload.get("causal_margin", 0.15),
        min_feedback_gain=payload.get("min_feedback_gain", 0.05),
        seed=payload.get("seed", 17),
    )


def main() -> None:
    args = build_parser().parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)

    experiment_name = config.get("experiment", {}).get("name", config_path.stem)
    output_dir = resolve_path(config.get("experiment", {}).get("output_dir", f"outputs/{experiment_name}"))
    data_cfg = config.get("data", {})
    baseline_cfg = config.get("baseline", {})
    runtime_cfg = config.get("runtime", {})
    pipeline_cfg = pipeline_config_from_dict(
        {
            **config.get("pipeline", {}),
            "seed": config.get("experiment", {}).get("seed", 17),
        }
    )

    dataset_path = resolve_path(data_cfg["dataset_path"])
    limit = args.limit if args.limit is not None else data_cfg.get("limit")
    samples = load_samples_auto(dataset_path, limit=limit)
    runner = build_baseline(baseline_cfg.get("name", "causal_ttl_mock"), config=pipeline_cfg)
    results = runner.run(samples)
    summary = summarize_results(results)

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": {
            "name": experiment_name,
            "config_path": str(config_path),
            "baseline": baseline_cfg.get("name", "causal_ttl_mock"),
            "dataset_path": str(dataset_path),
            "limit": limit,
        },
        "pipeline_config": asdict(pipeline_cfg),
        "summary": summary,
        "results": serialize_results(results),
    }

    if runtime_cfg.get("write_predictions_json", True):
        write_json_report(output_dir / "report.json", payload)
    if runtime_cfg.get("write_predictions_csv", True):
        write_csv_report(output_dir / "predictions.csv", results)

    print(f"Experiment: {experiment_name}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
