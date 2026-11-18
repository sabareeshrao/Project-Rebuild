"""Query-key building blocks for causal self-attention."""

import torch


def pairwise_dot_scores(queries: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Match each query with each key: [B,T,D] x [B,S,D] -> [B,T,S]."""
    if (
        queries.ndim != 3 or keys.ndim != 3
        or queries.numel() == 0 or keys.numel() == 0
        or not queries.is_floating_point() or not keys.is_floating_point()
        or queries.shape[0] != keys.shape[0]
        or queries.shape[-1] != keys.shape[-1]
    ):
        raise ValueError("Expected floating-point queries [B,T,D] and keys [B,S,D] with matching B,D")
    return queries @ keys.transpose(-2, -1)
