"""Small tensor helpers used to learn GPT data flow."""

from collections.abc import Sequence

import torch


def token_batch(rows: Sequence[Sequence[int]]) -> torch.Tensor:
    """Turn equal-length token ID sequences into a [batch, time] tensor."""
    batch = torch.tensor(rows, dtype=torch.long)
    if batch.ndim != 2 or batch.numel() == 0:
        raise ValueError("Expected a non-empty [batch, time] token array")
    return batch


def next_token_pairs(batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Make inputs and one-step-ahead targets for causal language modeling."""
    if batch.ndim != 2 or batch.size(1) < 2:
        raise ValueError("Expected a [batch, time] tensor with at least 2 tokens")
    return batch[:, :-1], batch[:, 1:]
