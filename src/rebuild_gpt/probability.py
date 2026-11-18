"""Numerically stable probability operations used in language modeling."""

import torch


def stable_softmax(logits: torch.Tensor) -> torch.Tensor:
    """Turn scores into probabilities along the final (vocabulary) axis."""
    if logits.ndim < 1 or logits.numel() == 0 or not logits.is_floating_point():
        raise ValueError("logits must be a non-empty floating-point tensor")
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    weights = shifted.exp()
    return weights / weights.sum(dim=-1, keepdim=True)


def next_token_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean negative log-probability of the correct token at each position."""
    if logits.ndim != 3 or logits.numel() == 0 or not logits.is_floating_point():
        raise ValueError("logits must be non-empty floating-point [batch, time, vocab]")
    if targets.shape != logits.shape[:2] or targets.dtype != torch.long:
        raise ValueError("targets must be torch.long with shape [batch, time]")
    if (targets < 0).any() or (targets >= logits.shape[-1]).any():
        raise ValueError("target token ID outside vocabulary")

    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    correct_token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1))
    return -correct_token_log_probs.mean()
