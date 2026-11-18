"""Small tensor helpers used to learn GPT data flow."""

from collections.abc import Sequence

import torch


def token_batch(rows: Sequence[Sequence[int]]) -> torch.Tensor:
    """Turn equal-length token ID sequences into a [batch, time] tensor."""
    batch = torch.tensor(rows, dtype=torch.long)
    if batch.ndim != 2 or batch.numel() == 0:
        raise ValueError("Expected a non-empty [batch, time] token array")
    return batch
