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
