from __future__ import annotations

from dataclasses import dataclass, field

from ..config import FrontDoorConfig


@dataclass(slots=True)
class SupervisionExportConfig:
    knowledge_in_context: bool = False
    supervise_reasoning: bool = True
    causal_path_format: bool = True
    include_causal_loss_samples: bool = True
    cloud_budget_ratio: float = 1.0
    frontdoor: FrontDoorConfig = field(default_factory=FrontDoorConfig)


def config_from_mode(
    mode: str,
    cloud_budget_ratio: float = 1.0,
    frontdoor: FrontDoorConfig | None = None,
) -> SupervisionExportConfig:
    normalized_mode = mode.strip().lower()
    frontdoor_config = frontdoor or FrontDoorConfig()
    if normalized_mode == "fact":
        return SupervisionExportConfig(
            knowledge_in_context=True,
            supervise_reasoning=True,
            causal_path_format=False,
            include_causal_loss_samples=False,
            cloud_budget_ratio=cloud_budget_ratio,
            frontdoor=frontdoor_config,
        )
    if normalized_mode == "v6":
        return SupervisionExportConfig(
            knowledge_in_context=True,
            supervise_reasoning=False,
            causal_path_format=False,
            include_causal_loss_samples=True,
            cloud_budget_ratio=cloud_budget_ratio,
            frontdoor=frontdoor_config,
        )
    if normalized_mode == "v7":
        return SupervisionExportConfig(
            knowledge_in_context=False,
            supervise_reasoning=True,
            causal_path_format=True,
            include_causal_loss_samples=True,
            cloud_budget_ratio=cloud_budget_ratio,
            frontdoor=frontdoor_config,
        )
    raise ValueError(f"Unsupported export mode `{mode}`. Supported modes: fact, v6, v7")
