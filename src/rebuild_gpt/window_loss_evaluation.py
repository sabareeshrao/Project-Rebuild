"""Evaluate each sampled window's mean next-token loss separately.

Per-window scalar losses allow stratified start-region reweighting.
A per-batch mean cannot reveal how much of that mean came from each
region when the batch contains starts drawn from several regions.
"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from rebuild_gpt.eval_index_plan import EvaluationIndexPlan
from rebuild_gpt.gpt import GPTLanguageModel


@dataclass(frozen=True)
class WindowLossEvaluation:
    starts: tuple[int, ...]
    per_window_losses: tuple[float, ...]
    batch_means: tuple[float, ...]

    @property
    def sampled_window_mean(self) -> float:
        return sum(self.per_window_losses) / len(self.per_window_losses)


def collect_window_losses(
    model: GPTLanguageModel,
    plan: EvaluationIndexPlan,
) -> WindowLossEvaluation:
    """Return one loss per sampled start in the original sampled order.

    Each window is scored using the average of its T next-token
    cross-entropies. The model is placed into eval mode temporarily;
    dropout is disabled and its original train/eval mode is restored.
    No gradients are computed or optimizer state modified.
    """
    if not isinstance(plan, EvaluationIndexPlan) or not plan.starts:
        raise ValueError("plan must contain a nonempty EvaluationIndexPlan")
    if plan.block_size > model.config.block_size:
        raise ValueError("block_size exceeds the model context window")
    starts = tuple(int(v) for batch in plan.starts for v in batch)
    if not starts:
        raise ValueError("plan must contain sampled window starts")
    device = next(model.parameters()).device
    was_training = model.training
    scores = []
    batch_means = []
    try:
        model.eval()
        with torch.no_grad():
            for inputs, targets in plan.batches():
                logits = model(inputs.to(device))
                targets = targets.to(device)
                token_losses = F.cross_entropy(
                    logits.transpose(1, 2), targets, reduction="none",
                )
                per_window = token_losses.mean(dim=1).cpu().tolist()
                scores.extend(float(value) for value in per_window)
                batch_means.append(sum(per_window) / len(per_window))
    finally:
        model.train(was_training)
    if len(scores) != len(starts):
        raise RuntimeError("per-window loss count does not match sampled starts")
    return WindowLossEvaluation(
        starts=starts,
        per_window_losses=tuple(scores),
        batch_means=tuple(batch_means),
    )
