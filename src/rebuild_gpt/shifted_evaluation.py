"""Seeded, phase-shifted disjoint windows for held-out GPT evaluation.

Unlike fixed-origin disjoint tiles, the lattice begins at a random
feasible phase. The policy changes the sampling distribution relative
to uniform start-with-replacement; it is NOT an unbiased estimator.
"""

import torch

from rebuild_gpt.eval_index_plan import EvaluationIndexPlan


def feasible_lattice_phases(
    *, corpus_token_count: int, block_size: int, window_count: int,
) -> int:
    """Number of offsets in [0,T] that can hold window_count full tiles."""
    for label, value in (
        ("corpus_token_count", corpus_token_count),
        ("block_size", block_size),
        ("window_count", window_count),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    width = block_size + 1
    capacity = corpus_token_count // width
    if window_count > capacity:
        raise ValueError(
            f"shifted disjoint windows require {window_count} tiles but "
            f"corpus has only {capacity} nonoverlapping tiles"
        )
    return min(width, corpus_token_count - window_count * width + 1)


def prepare_shifted_evaluation_index_plan(
    tokens: torch.Tensor,
    *,
    batch_size: int,
    block_size: int,
    eval_batches: int = 4,
    seed: int = 0,
) -> EvaluationIndexPlan:
    """Select one feasible phase and draw distinct tiles using local RNG.

    Feasible phases depend on the requested count: a near-capacity
    request might force phase zero. When that feasible set changes,
    increasing eval_batches need not preserve the old sampled prefix.
    """
    for label, value in (
        ("batch_size", batch_size), ("block_size", block_size),
        ("eval_batches", eval_batches),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (
        not isinstance(tokens, torch.Tensor)
        or tokens.ndim != 1 or tokens.dtype != torch.long
    ):
        raise ValueError("tokens must be a flat torch.long stream")
    cpu_tokens = tokens.cpu()
    requested = batch_size * eval_batches
    phase_count = feasible_lattice_phases(
        corpus_token_count=cpu_tokens.numel(),
        block_size=block_size, window_count=requested,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    phase = int(torch.randint(0, phase_count, (1,), generator=generator).item())
    width = block_size + 1
    eligible_tiles = (cpu_tokens.numel() - phase) // width
    indices = torch.randperm(eligible_tiles, generator=generator)[:requested]
    selected = phase + indices * width
    starts = tuple(
        selected[i:i + batch_size].clone()
        for i in range(0, requested, batch_size)
    )
    return EvaluationIndexPlan(
        tokens=cpu_tokens, starts=starts,
        batch_size=batch_size, block_size=block_size, seed=seed,
    )
