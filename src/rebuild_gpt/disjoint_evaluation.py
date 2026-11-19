"""Sample nonoverlapping held-out token windows on a fixed lattice.

This alternative is explicit because disjoint windows are NOT drawn
from the same distribution as the original every-start sampler.
"""

import torch

from rebuild_gpt.eval_index_plan import EvaluationIndexPlan


def prepare_disjoint_evaluation_index_plan(
    tokens: torch.Tensor,
    *,
    batch_size: int,
    block_size: int,
    eval_batches: int = 4,
    seed: int = 0,
) -> EvaluationIndexPlan:
    """Draw distinct T+1-token tiles, shuffled without replacement.

    Tile starts are 0, T+1, 2(T+1), ... . Each tile includes both
    next-token contexts and targets. The last short tail is unused,
    and no tile repeats within one plan. Different seeds can still
    choose overlapping *sets* of tiles across plans.
    """
    for name, value in (
        ("batch_size", batch_size),
        ("block_size", block_size),
        ("eval_batches", eval_batches),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (
        not isinstance(tokens, torch.Tensor)
        or tokens.ndim != 1 or tokens.dtype != torch.long
    ):
        raise ValueError("tokens must be a flat torch.long stream")
    cpu_tokens = tokens.cpu()
    tile_width = block_size + 1
    tile_count = cpu_tokens.numel() // tile_width
    requested = batch_size * eval_batches
    if requested > tile_count:
        raise ValueError(
            f"disjoint windows require {requested} tiles but corpus has "
            f"only {tile_count} nonoverlapping tiles"
        )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    chosen = torch.randperm(tile_count, generator=generator)[:requested] * tile_width
    starts = tuple(
        chosen[i:i + batch_size].clone()
        for i in range(0, requested, batch_size)
    )
    return EvaluationIndexPlan(
        tokens=cpu_tokens, starts=starts,
        batch_size=batch_size, block_size=block_size, seed=seed,
    )
