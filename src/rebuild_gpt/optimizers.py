"""AdamW parameter groups following original nanoGPT's tensor-rank rule."""

import inspect
import math

import torch
from torch import nn


def configure_adamw(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    device_type: str = "cpu",
) -> torch.optim.AdamW:
    """Decay 2D+ trainable weights, but exempt 1D biases and LayerNorm weights.

    Parameters are enumerated once via named_parameters(), so tied token/output
    embeddings are not accidentally inserted into the optimizer twice.
    """
    if (
        isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float))
        or not math.isfinite(learning_rate) or learning_rate <= 0
    ):
        raise ValueError("learning_rate must be finite and positive")
    if (
        isinstance(weight_decay, bool) or not isinstance(weight_decay, (int, float))
        or not math.isfinite(weight_decay) or weight_decay < 0
    ):
        raise ValueError("weight_decay must be finite and nonnegative")
    if (
        not isinstance(betas, (tuple, list)) or len(betas) != 2
        or any(isinstance(b, bool) or not isinstance(b, (float, int))
               or not math.isfinite(b) or not 0 <= b < 1 for b in betas)
    ):
        raise ValueError("betas must contain two finite values in [0,1)")
    if device_type not in ("cpu", "cuda"):
        raise ValueError("device_type must be cpu or cuda")
    trainable = [param for param in model.parameters() if param.requires_grad]
    if not trainable:
        raise ValueError("model has no trainable parameters")
    decay = [param for param in trainable if param.ndim >= 2]
    no_decay = [param for param in trainable if param.ndim < 2]
    groups = [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused = device_type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters
    kwargs = {"fused": True} if fused else {}
    return torch.optim.AdamW(groups, lr=learning_rate, betas=tuple(betas), **kwargs)
