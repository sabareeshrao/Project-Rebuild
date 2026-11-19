"""Prepare repeatable CPU evaluation batches once, for reuse across steps.

A prepared plan keeps a copy of each sampled input and target batch.
This trades host RAM for avoiding repeated random indexing at each
validation event. The original corpus and model parameters are unchanged.
"""

from dataclasses import dataclass

import torch

from rebuild_gpt.data import sample_batch


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
