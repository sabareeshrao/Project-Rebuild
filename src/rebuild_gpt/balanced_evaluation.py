"""Seeded region-balanced validation starts with local CPU randomness.

This explicitly changes the sampling distribution: each region of
*eligible window starts* receives nearly the same number of draws.
Starts are sampled with replacement within each region; sampled token
windows can still overlap, even when all regions receive starts.
"""

import torch

from rebuild_gpt.eval_index_plan import EvaluationIndexPlan


def start_region_bounds(
    *, corpus_token_count: int, block_size: int, region_count: int,
) -> tuple[tuple[int, int], ...]:
    """Partition the valid start indices into nonempty, adjacent regions."""
    for name, value in (
        ("corpus_token_count", corpus_token_count),
        ("block_size", block_size),
        ("region_count", region_count),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    valid_starts = corpus_token_count - block_size
    if valid_starts < 1:
        raise ValueError("corpus_token_count must exceed block_size")
    if region_count > valid_starts:
        raise ValueError("region_count exceeds number of valid window starts")
    return tuple(
        (i * valid_starts // region_count,
         (i + 1) * valid_starts // region_count)
        for i in range(region_count)
    )


def prepare_balanced_evaluation_index_plan(
    tokens: torch.Tensor,
    *,
    batch_size: int,
    block_size: int,
    eval_batches: int = 4,
    seed: int = 0,
    region_count: int = 4,
) -> EvaluationIndexPlan:
    """Sample approximately equal *start counts* in every start region.

    Each region receives either floor(W/R) or ceil(W/R) draws, where
    W=batch_size*eval_batches and R=region_count. Sampling uses a
    private CPU generator. Budget changes can change every region's
    allocation, so increasing eval_batches need not preserve a prefix.
    """
    for name, value in (
        ("batch_size", batch_size), ("block_size", block_size),
        ("eval_batches", eval_batches), ("region_count", region_count),
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
    bounds = start_region_bounds(
        corpus_token_count=cpu_tokens.numel(),
        block_size=block_size, region_count=region_count,
    )
    requested = batch_size * eval_batches
    if requested < region_count:
        raise ValueError(
            "balanced sampling requires at least one window per region_count"
        )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    base, extras = divmod(requested, region_count)
    selected = torch.cat([
        torch.randint(
            low, high, (base + (index < extras),),
            generator=generator, dtype=torch.long,
        )
        for index, (low, high) in enumerate(bounds)
    ])
    starts = tuple(
        selected[index:index + batch_size].clone()
        for index in range(0, requested, batch_size)
    )
    return EvaluationIndexPlan(
        tokens=cpu_tokens, starts=starts, batch_size=batch_size,
        block_size=block_size, seed=seed,
    )
