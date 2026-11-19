"""Create small, deterministic training-workload experiment grids."""

from itertools import product

import torch

from rebuild_gpt.workload_scaling import WorkloadEstimate, estimate_workload


def parse_positive_sizes(spec: str, *, label: str) -> tuple[int, ...]:
    """Parse a nonempty, duplicate-free comma-separated list of integers."""
    if not isinstance(label, str) or not label.strip():
        raise ValueError("label must be a nonempty string")
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError(f"{label} must be a nonempty comma-separated list")
    parts = spec.split(",")
    try:
        sizes = tuple(int(part.strip()) for part in parts)
    except ValueError as exc:
        raise ValueError(f"{label} must contain only integers") from exc
    if any(value < 1 for value in sizes) or len(set(sizes)) != len(sizes):
        raise ValueError(f"{label} must contain unique positive integers")
    return sizes


def build_scaling_grid(
    batch_sizes: tuple[int, ...],
    context_lengths: tuple[int, ...],
    depths: tuple[int, ...],
    *,
    n_head: int = 4,
    gradient_accumulation_steps: int = 1,
    dtype: torch.dtype = torch.float32,
    max_points: int = 64,
) -> tuple[WorkloadEstimate, ...]:
    """Combine shapes in caller-defined order; bound experiment count."""
    for label, sizes in (
        ("batch_sizes", batch_sizes),
        ("context_lengths", context_lengths),
        ("depths", depths),
    ):
        if not isinstance(sizes, tuple) or not sizes:
            raise ValueError(f"{label} must be a nonempty tuple")
        if any(type(value) is not int or value < 1 for value in sizes):
            raise ValueError(f"{label} must contain positive integers")
        if len(set(sizes)) != len(sizes):
            raise ValueError(f"{label} must not contain duplicates")
    if type(max_points) is not int or max_points < 1:
        raise ValueError("max_points must be a positive integer")
    count = len(batch_sizes) * len(context_lengths) * len(depths)
    if count > max_points:
        raise ValueError(
            f"Grid has {count} points, exceeding max_points={max_points}"
        )
    return tuple(
        estimate_workload(
            batch_size=batch, context_length=context,
            n_head=n_head, n_layer=depth,
            gradient_accumulation_steps=gradient_accumulation_steps,
            dtype=dtype,
        )
        for batch, context, depth in product(
            batch_sizes, context_lengths, depths
        )
    )
