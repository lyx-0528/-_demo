from __future__ import annotations

import os
from typing import Optional

from .base import BaseTeacher
from .heuristic import HeuristicCausalTeacher
from .openai_api import OpenAICompatibleTeacher


def build_teacher(
    backend: str = "heuristic",
    *,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_new_tokens: int = 512,
    cache_dir: Optional[str] = None,
) -> BaseTeacher:
    normalized_backend = backend.strip().lower()
    if normalized_backend == "heuristic":
        return HeuristicCausalTeacher(cache_dir=cache_dir)
    if normalized_backend != "api":
        raise ValueError(f"Unsupported teacher backend `{backend}`. Supported backends: heuristic, api")

    resolved_api_base = api_base or os.getenv("TEACHER_API_BASE") or os.getenv("OPENAI_API_BASE")
    resolved_api_key = api_key or os.getenv("TEACHER_API_KEY") or os.getenv("OPENAI_API_KEY")
    resolved_model = model or os.getenv("TEACHER_MODEL") or os.getenv("TEACHER_MODEL_NAME")

    if not resolved_api_base:
        raise ValueError("Missing teacher API base. Pass `api_base` or set TEACHER_API_BASE.")
    if not resolved_model:
        raise ValueError("Missing teacher model. Pass `model` or set TEACHER_MODEL.")

    return OpenAICompatibleTeacher(
        api_base=resolved_api_base,
        api_key=resolved_api_key,
        model=resolved_model,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        cache_dir=cache_dir,
    )
