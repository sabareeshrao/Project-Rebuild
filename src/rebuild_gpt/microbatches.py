"""Divide one logical batch into smaller, ordered microbatches.

A microbatch split is a view of existing tensors: it changes *how*
examples are processed, not the examples themselves.
"""

from collections.abc import Iterator

import torch


def microbatch_sizes(total_batch_size: int, microbatch_size: int) -> tuple[int, ...]:
    """Return nonempty chunk sizes, including a smaller final chunk."""
    for label, value in (
        ("total_batch_size", total_batch_size),
        ("microbatch_size", microbatch_size),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    return tuple(
        min(microbatch_size, total_batch_size - start)
        for start in range(0, total_batch_size, microbatch_size)
    )


def split_microbatches(
    inputs: torch.Tensor, targets: torch.Tensor, *,
    microbatch_size: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Split corresponding input/target rows without copying data.

    Targets can be [B] for classification or [B,T] for a GPT; only
    their leading batch dimension must match the inputs.
    """
    if (
        not isinstance(inputs, torch.Tensor)
        or not isinstance(targets, torch.Tensor)
        or inputs.ndim < 1 or targets.ndim < 1
        or inputs.size(0) < 1
        or inputs.size(0) != targets.size(0)
    ):
        raise ValueError("inputs and targets must have the same nonempty batch axis")
    sizes = microbatch_sizes(inputs.size(0), microbatch_size)
    result = []
    start = 0
    for size in sizes:
        stop = start + size
        result.append((inputs[start:stop], targets[start:stop]))
        start = stop
    return result


def iter_weighted_microbatches(
    inputs: torch.Tensor, targets: torch.Tensor, *,
    microbatch_size: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, float]]:
    """Yield one view at a time, weighted by its number of targets.

    The iterator is deliberately lazy: a caller may transfer only the
    current chunk to an accelerator, backpropagate, then release it.
    Weights add to 1.0, including an uneven final microbatch.
    """
    if (
        not isinstance(inputs, torch.Tensor)
        or not isinstance(targets, torch.Tensor)
        or inputs.ndim < 1 or targets.ndim < 1
        or inputs.size(0) < 1
        or inputs.size(0) != targets.size(0)
        or targets.numel() == 0
    ):
        raise ValueError("inputs and targets must have the same nonempty batch axis")
    sizes = microbatch_sizes(inputs.size(0), microbatch_size)
    total_targets = targets.numel()
    start = 0
    for size in sizes:
        end = start + size
        yield (
            inputs[start:end],
            targets[start:end],
            targets[start:end].numel() / total_targets,
        )
        start = end
