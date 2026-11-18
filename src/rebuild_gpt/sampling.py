"""Small, testable transformations for GPT next-token sampling."""

import math

import torch


def validate_temperature(temperature: float) -> float:
    """Require a finite positive value; zero is not an implicit greedy mode."""
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("temperature must be finite and positive")
    return float(temperature)


def scale_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Rescale real-valued [batch, vocab] logits before applying softmax."""
    t = validate_temperature(temperature)
    if logits.ndim != 2 or logits.numel() == 0 or not logits.is_floating_point():
        raise ValueError("logits must be a non-empty float [batch, vocab] tensor")
    return logits / t


def validate_top_k(top_k: int | None) -> int | None:
    """Allow None or a strictly positive integer count of candidate tokens."""
    if top_k is None:
        return None
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer or None")
    return top_k


def filter_top_k(logits: torch.Tensor, top_k: int | None) -> torch.Tensor:
    """Mask logits below the kth largest value, preserving ties at the cutoff.

    Returns a new tensor and never edits the input logits in place.
    """
    k = validate_top_k(top_k)
    if logits.ndim != 2 or logits.numel() == 0 or not logits.is_floating_point():
        raise ValueError("logits must be a non-empty float [batch, vocab] tensor")
    if k is None or k >= logits.size(-1):
        return logits
    threshold = torch.topk(logits, k, dim=-1).values[:, [-1]]
    return logits.masked_fill(logits < threshold, float("-inf"))
