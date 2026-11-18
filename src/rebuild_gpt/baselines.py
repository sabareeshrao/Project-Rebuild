"""Simple non-trainable baselines to understand model predictions."""

import torch


def uniform_logits(inputs: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Give every token the same score at each [batch, time] position."""
    if inputs.ndim != 2 or inputs.numel() == 0 or inputs.dtype != torch.long:
        raise ValueError("inputs must be a non-empty [batch, time] long tensor")
    if vocab_size < 1:
        raise ValueError("vocab_size must be positive")
    return torch.zeros((*inputs.shape, vocab_size), device=inputs.device)
