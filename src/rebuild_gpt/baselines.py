"""Simple non-trainable baselines to understand model predictions."""

import torch


def uniform_logits(inputs: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Give every token the same score at each [batch, time] position."""
    if inputs.ndim != 2 or inputs.numel() == 0 or inputs.dtype != torch.long:
        raise ValueError("inputs must be a non-empty [batch, time] long tensor")
    if vocab_size < 1:
        raise ValueError("vocab_size must be positive")
    return torch.zeros((*inputs.shape, vocab_size), device=inputs.device)


def unigram_probabilities(tokens: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Estimate next-token probabilities using training token frequencies.

    Add-one smoothing keeps unseen tokens at nonzero probability.
    This baseline ignores context: every position has the same distribution.
    """
    if tokens.ndim != 1 or tokens.numel() == 0 or tokens.dtype != torch.long:
        raise ValueError("tokens must be a non-empty flat long tensor")
    if vocab_size < 1:
        raise ValueError("vocab_size must be positive")
    if (tokens < 0).any() or (tokens >= vocab_size).any():
        raise ValueError("token ID outside vocabulary")
    counts = torch.bincount(tokens, minlength=vocab_size).float()
    return (counts + 1) / (counts.sum() + vocab_size)
