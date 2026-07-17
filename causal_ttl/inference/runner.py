from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Optional

from ..schema import MedSample
from ..teachers import BaseTeacher
from .config import InferenceRuntimeConfig
from .generate import extract_final_answer, generate_direct, interactive_generate
from .prompts import build_inference_prompt
from .runtime import load_model_and_tokenizer


def predict_samples(
    samples: list[MedSample],
    *,
    config: InferenceRuntimeConfig,
    teacher: Optional[BaseTeacher] = None,
) -> list[dict[str, Any]]:
    model, tokenizer = load_model_and_tokenizer(config)
    rows: list[dict[str, Any]] = []

    for sample in samples:
        payload = teacher.generate_causal_sample(sample) if teacher is not None else None
        prompt = build_inference_prompt(
            sample,
            mode=config.mode,
            teacher_payload=payload,
            knowledge_at_inference=config.knowledge_at_inference,
        )

        if config.mode == "v7":
            if teacher is None:
                raise ValueError("Mode `v7` requires a teacher for interactive inference.")
            full_predict, interactions = interactive_generate(
                model,
                tokenizer,
                sample,
                prompt,
                teacher=teacher,
                config=config,
            )
        else:
            full_predict = generate_direct(
                model,
                tokenizer,
                prompt,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                do_sample=config.do_sample,
                top_p=config.top_p,
            )
            interactions = []

        rows.append(
            {
                "sample_id": sample.sample_id,
                "source": sample.source,
                "question": sample.question,
                "gold_answer": sample.gold_answer,
                "prompt": prompt,
                "predict": extract_final_answer(full_predict),
                "full_predict": full_predict,
                "cloud_calls": len(interactions),
                "interactions": interactions,
            }
        )

    return rows


def write_predictions(rows: list[dict[str, Any]], path: Path, config: InferenceRuntimeConfig) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "config": {
            **asdict(config),
            "predictions_path": str(config.predictions_path),
            "adapter_path": str(config.adapter_path) if config.adapter_path is not None else None,
        },
        "num_samples": len(rows),
        "avg_cloud_calls": (sum(row["cloud_calls"] for row in rows) / len(rows)) if rows else 0.0,
    }
    summary_path = path.with_name(path.stem + "_summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
