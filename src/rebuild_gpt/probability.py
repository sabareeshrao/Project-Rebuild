"""Numerically stable probability operations used in language modeling."""

import torch


def stable_softmax(logits: torch.Tensor) -> torch.Tensor:
    """Turn scores into probabilities along the final (vocabulary) axis."""
    if logits.ndim < 1 or logits.numel() == 0 or not logits.is_floating_point():
        raise ValueError("logits must be a non-empty floating-point tensor")
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    weights = shifted.exp()
    return weights / weights.sum(dim=-1, keepdim=True)
