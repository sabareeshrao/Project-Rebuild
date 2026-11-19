"""Conservative host-memory accounting for prepared next-token batches.

Each sampled input/target pair has two logical [B,T] int64 tensors.
They currently share storage for an underlying [B,T+1] window, so
the logical-byte total is an intentionally conservative budget bound,
not an exact measurement of physical RSS.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EvaluationCacheBudget:
    batch_size: int
    block_size: int
    eval_batches: int
    split_count: int
    bytes_per_split: int
    total_bytes: int

    @property
    def total_mib(self) -> float:
        return self.total_bytes / (1024 * 1024)


def estimate_evaluation_cache_budget(
    *, batch_size: int, block_size: int,
    eval_batches: int, split_count: int = 2,
) -> EvaluationCacheBudget:
    """Return an upper bound on retained logical sampled-token bytes."""
    for label, value in (
        ("batch_size", batch_size), ("block_size", block_size),
        ("eval_batches", eval_batches), ("split_count", split_count),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{label} must be a positive integer")
    token_bytes = torch.empty((), dtype=torch.long).element_size()
    per_split = 2 * batch_size * block_size * eval_batches * token_bytes
    return EvaluationCacheBudget(
        batch_size=batch_size, block_size=block_size,
        eval_batches=eval_batches, split_count=split_count,
        bytes_per_split=per_split, total_bytes=per_split * split_count,
    )


def enforce_evaluation_cache_budget(
    required_bytes: int, max_bytes: int | None,
) -> None:
    """Reject an oversized plan *before* random drawing or tensor allocation."""
    if type(required_bytes) is not int or required_bytes < 1:
        raise ValueError("required_bytes must be a positive integer")
    if max_bytes is not None:
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer or None")
        if required_bytes > max_bytes:
            raise ValueError(
                f"evaluation cache needs {required_bytes} logical bytes, "
                f"exceeding max_bytes={max_bytes}"
            )
