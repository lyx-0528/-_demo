from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class InferenceRuntimeConfig:
    model_name_or_path: str
    predictions_path: Path
    adapter_path: Optional[Path] = None
    mode: str = "v7"
    max_length: int = 2048
    max_new_tokens: int = 512
    temperature: float = 0.0
    do_sample: bool = False
    top_p: float = 1.0
    trust_remote_code: bool = True
    knowledge_at_inference: bool = True
    chunk_size: int = 64
    max_cloud_calls: int = 3
    cloud_budget_ratio: float = 1.0
