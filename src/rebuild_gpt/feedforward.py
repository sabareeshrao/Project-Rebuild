"""A per-token feed-forward network for transformer blocks."""

import torch
from torch import nn


class FeedForward(nn.Module):
    """C -> 4C -> GELU -> C with training-only output dropout."""

    def __init__(self, n_embd: int, dropout: float = 0.0) -> None:
        super().__init__()
        if n_embd < 1:
            raise ValueError("n_embd must be positive")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1")
        self.n_embd = n_embd
        self.expand = nn.Linear(n_embd, 4 * n_embd)
        self.activation = nn.GELU()
        self.project = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if (
            hidden.ndim != 3 or hidden.numel() == 0
            or not hidden.is_floating_point() or hidden.size(-1) != self.n_embd
        ):
            raise ValueError("hidden must be floating-point [B,T,n_embd]")
        return self.dropout(self.project(self.activation(self.expand(hidden))))
