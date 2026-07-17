from __future__ import annotations

from .config import TrainingRuntimeConfig
from .dataset import IGNORE_INDEX, TASK_TYPE_IDS


class SupervisionDataCollator:
    def __init__(self, tokenizer, model, label_pad_token_id: int, pad_to_multiple_of: int | None = None) -> None:
        from transformers import DataCollatorForSeq2Seq

        self._collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=pad_to_multiple_of,
        )

    def __call__(self, features: list[dict[str, object]]) -> dict[str, object]:
        import torch

        stripped_features: list[dict[str, object]] = []
        task_type_ids: list[int] = []
        sample_weights: list[float] = []
        for feature in features:
            row = dict(feature)
            task_type_ids.append(int(row.pop("task_type_id", 0)))
            sample_weights.append(float(row.pop("sample_weight", 1.0)))
            row.pop("record_id", None)
            row.pop("task_type", None)
            stripped_features.append(row)

        batch = self._collator(stripped_features)
        batch["task_type_id"] = torch.tensor(task_type_ids, dtype=torch.long)
        batch["sample_weight"] = torch.tensor(sample_weights, dtype=torch.float32)
        return batch


def build_weighted_trainer_class():
    import torch
    import torch.nn as nn
    from transformers import Trainer

    class _WeightedCausalTrainer(Trainer):
        def __init__(self, runtime_config: TrainingRuntimeConfig, **kwargs) -> None:
            super().__init__(**kwargs)
            self.runtime_config = runtime_config

        def _per_sample_ce(self, logits: "torch.Tensor", labels: "torch.Tensor") -> "torch.Tensor":
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_mask = shift_labels.ne(IGNORE_INDEX)
            safe_labels = shift_labels.masked_fill(~loss_mask, 0)
            token_loss = nn.CrossEntropyLoss(reduction="none")(
                shift_logits.view(-1, shift_logits.size(-1)),
                safe_labels.view(-1),
            ).view(shift_labels.size())
            return (token_loss * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1).clamp_min(1)

        def _task_weights(self, task_type_ids: "torch.Tensor", sample_loss: "torch.Tensor") -> "torch.Tensor":
            weights = torch.full_like(sample_loss, float(self.runtime_config.answer_loss_weight))
            weights = torch.where(
                task_type_ids.eq(TASK_TYPE_IDS["counterfactual"]),
                torch.full_like(sample_loss, float(self.runtime_config.counterfactual_loss_weight)),
                weights,
            )
            weights = torch.where(
                task_type_ids.eq(TASK_TYPE_IDS["policy"]),
                torch.full_like(sample_loss, float(self.runtime_config.policy_loss_weight)),
                weights,
            )
            return weights

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            task_type_ids = inputs.pop("task_type_id", None)
            sample_weight = inputs.pop("sample_weight", None)
            outputs = model(**inputs)
            sample_loss = self._per_sample_ce(outputs.logits, inputs["labels"])

            if task_type_ids is None:
                task_type_ids = torch.zeros_like(sample_loss, dtype=torch.long)
            else:
                task_type_ids = task_type_ids.to(sample_loss.device).view(-1)

            weights = self._task_weights(task_type_ids, sample_loss)
            if sample_weight is not None and self.runtime_config.use_sample_weight:
                weights = weights * sample_weight.to(sample_loss.device).view(-1)

            if self.runtime_config.normalize_weighted_loss:
                total_loss = (sample_loss * weights).sum() / weights.sum().clamp_min(1e-8)
            else:
                total_loss = (sample_loss * weights).mean()

            return (total_loss, outputs) if return_outputs else total_loss

    return _WeightedCausalTrainer
