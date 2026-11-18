"""Educational LayerNorm calculations for the last feature dimension."""

import math

import torch


def normalize_features(hidden: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Normalize every token's features, independently of other tokens."""
    if hidden.numel() == 0 or not hidden.is_floating_point():
        raise ValueError("hidden must be a non-empty floating-point tensor")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    mean = hidden.mean(dim=-1, keepdim=True)
    variance = ((hidden - mean) ** 2).mean(dim=-1, keepdim=True)
    return (hidden - mean) * torch.rsqrt(variance + eps)
