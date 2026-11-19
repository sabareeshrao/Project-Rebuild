"""Prepare repeatable CPU evaluation batches once, for reuse across steps.

A prepared plan keeps a copy of each sampled input and target batch.
This trades host RAM for avoiding repeated random indexing at each
validation event. The original corpus and model parameters are unchanged.
"""

from dataclasses import dataclass

import torch

from rebuild_gpt.data import sample_batch
from rebuild_gpt.eval_cache_budget import (
    enforce_evaluation_cache_budget, estimate_evaluation_cache_budget,
)


@dataclass(frozen=True)
class PreparedEvaluationBatches:
    batches: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    batch_size: int
    block_size: int
    seed: int

    @property
    def storage_bytes(self) -> int:
        """Bytes retained by sampled inputs and targets, not total process RAM."""
        return sum(
            x.numel() * x.element_size() + y.numel() * y.element_size()
            for x, y in self.batches
        )


def prepare_evaluation_batches(
    tokens: torch.Tensor,
    *,
    batch_size: int,
    block_size: int,
    eval_batches: int = 4,
    seed: int = 0,
    max_bytes: int | None = None,
) -> PreparedEvaluationBatches:
    """Match the seeded sample_batch sequence without touching global RNG.

    All batches remain on CPU so the same plan can evaluate a CPU or
    CUDA model. Model transfers occur only while evaluating the plan.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if type(block_size) is not int or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if type(eval_batches) is not int or eval_batches < 1:
        raise ValueError("eval_batches must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(tokens, torch.Tensor):
        raise ValueError("tokens must be a torch.Tensor")
    estimate = estimate_evaluation_cache_budget(
        batch_size=batch_size, block_size=block_size,
        eval_batches=eval_batches, split_count=1,
    )
    enforce_evaluation_cache_budget(estimate.total_bytes, max_bytes)
    cpu_tokens = tokens.cpu()
    rng = torch.Generator(device="cpu").manual_seed(seed)
    batches = tuple(
        sample_batch(
            cpu_tokens,
            batch_size=batch_size, block_size=block_size, generator=rng,
        )
        for _ in range(eval_batches)
    )
    return PreparedEvaluationBatches(
        batches=batches, batch_size=batch_size,
        block_size=block_size, seed=seed,
    )


def prepare_train_val_evaluation_batches(
    train_tokens: torch.Tensor,
    validation_tokens: torch.Tensor,
    *,
    batch_size: int,
    block_size: int,
    eval_batches: int = 4,
    seed: int = 0,
    max_bytes: int | None = None,
) -> tuple[PreparedEvaluationBatches, PreparedEvaluationBatches]:
    """Prepare both splits after checking the *combined* memory limit.

    The separate CPU generators match estimate_train_val_loss(seed),
    whose train stream uses seed and validation stream uses seed+1.
    A too-small budget fails before allocating either split's batches.
    """
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    estimate = estimate_evaluation_cache_budget(
        batch_size=batch_size, block_size=block_size,
        eval_batches=eval_batches, split_count=2,
    )
    enforce_evaluation_cache_budget(estimate.total_bytes, max_bytes)
    return (
        prepare_evaluation_batches(
            train_tokens, batch_size=batch_size,
            block_size=block_size, eval_batches=eval_batches, seed=seed,
        ),
        prepare_evaluation_batches(
            validation_tokens, batch_size=batch_size,
            block_size=block_size, eval_batches=eval_batches, seed=seed + 1,
        ),
    )
