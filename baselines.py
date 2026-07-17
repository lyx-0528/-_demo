from __future__ import annotations

from dataclasses import replace

from causal_ttl import CausalTTLPipeline, PipelineConfig


class BaselineRunner:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def run(self, samples):
        raise NotImplementedError


class StaticMockRunner(BaselineRunner):
    def run(self, samples):
        static_config = replace(self.config, min_feedback_gain=999.0)
        pipeline = CausalTTLPipeline(config=static_config)
        return pipeline.run_dataset(samples)


class CausalTTLMockRunner(BaselineRunner):
    def run(self, samples):
        pipeline = CausalTTLPipeline(config=self.config)
        return pipeline.run_dataset(samples)


BASELINE_REGISTRY = {
    "static_mock": StaticMockRunner,
    "causal_ttl_mock": CausalTTLMockRunner,
}


def build_baseline(name: str, config: PipelineConfig) -> BaselineRunner:
    if name not in BASELINE_REGISTRY:
        supported = ", ".join(sorted(BASELINE_REGISTRY))
        raise ValueError(f"Unsupported baseline `{name}`. Supported baselines: {supported}")
    return BASELINE_REGISTRY[name](config=config)
