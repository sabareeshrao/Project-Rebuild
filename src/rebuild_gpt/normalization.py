"""Educational LayerNorm calculations for the last feature dimension."""

import math

import torch
from torch import nn


def normalize_features(hidden: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Normalize every token's features, independently of other tokens."""
    if hidden.numel() == 0 or not hidden.is_floating_point():
        raise ValueError("hidden must be a non-empty floating-point tensor")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    mean = hidden.mean(dim=-1, keepdim=True)
    variance = ((hidden - mean) ** 2).mean(dim=-1, keepdim=True)
    return (hidden - mean) * torch.rsqrt(variance + eps)


class FeatureLayerNorm(nn.Module):
    """Normalize C features with learned per-feature scale and bias."""

    def __init__(self, n_embd: int, eps: float = 1e-5, *, bias: bool = True) -> None:
        super().__init__()
        if n_embd < 1:
            raise ValueError("n_embd must be positive")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.n_embd = n_embd
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd)) if bias else None

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 3 or hidden.size(-1) != self.n_embd:
            raise ValueError("hidden must have shape [B,T,n_embd]")
        normalized = normalize_features(hidden, self.eps) * self.weight
        return normalized if self.bias is None else normalized + self.bias
