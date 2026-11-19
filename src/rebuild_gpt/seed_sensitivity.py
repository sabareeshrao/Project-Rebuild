"""Measure how a fixed model's sampled validation mean varies by seed."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from rebuild_gpt.evaluation_samples import collect_split_losses
from rebuild_gpt.evaluation_stats import LossDispersion, summarize_losses
from rebuild_gpt.gpt import GPTLanguageModel


@dataclass(frozen=True)
class SeedSensitivity:
    seeds: tuple[int, ...]
    per_seed_means: tuple[float, ...]
    summary: LossDispersion
    eval_batches_per_seed: int


def evaluate_seed_sensitivity(
    model: GPTLanguageModel,
    tokens: torch.Tensor,
    *,
    seeds: Sequence[int],
    batch_size: int,
    block_size: int,
    eval_batches: int = 4,
) -> SeedSensitivity:
    """Evaluate unchanged model weights over independent seeded draws.

    Separate seeds define different batches, but overlapping windows
    and a single held-out corpus mean the observations are not
    statistically independent evidence about generalization.
    """
    if not isinstance(seeds, Sequence) or isinstance(seeds, (str, bytes)):
        raise ValueError("seeds must be a nonempty sequence of unique nonnegative integers")
    chosen = tuple(seeds)
    if (not chosen or any(type(seed) is not int or seed < 0 for seed in chosen)
            or len(set(chosen)) != len(chosen)):
        raise ValueError("seeds must be a nonempty sequence of unique nonnegative integers")
    if not isinstance(tokens, torch.Tensor):
        raise ValueError("tokens must be a torch.Tensor")
    cpu_tokens = tokens.cpu()
    per_seed_means = []
    for seed in chosen:
        losses = collect_split_losses(
            model, cpu_tokens, seed=seed, batch_size=batch_size,
            block_size=block_size, eval_batches=eval_batches,
        )
        per_seed_means.append(sum(losses) / len(losses))
    means = tuple(per_seed_means)
    return SeedSensitivity(
        seeds=chosen, per_seed_means=means,
        summary=summarize_losses(means),
        eval_batches_per_seed=eval_batches,
    )
