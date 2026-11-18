"""Query-key building blocks for causal self-attention."""

import torch
from torch import nn


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


class QueryKeyProjection(nn.Module):
    """Learn separate query and key vectors from each input token vector."""

    def __init__(self, n_embd: int, head_size: int) -> None:
        super().__init__()
        if n_embd < 1 or head_size < 1:
            raise ValueError("n_embd and head_size must be positive")
        self.n_embd = n_embd
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            hidden.ndim != 3 or hidden.numel() == 0
            or not hidden.is_floating_point() or hidden.size(-1) != self.n_embd
        ):
            raise ValueError("hidden must be floating-point [B,T,n_embd]")
        return self.query(hidden), self.key(hidden)
