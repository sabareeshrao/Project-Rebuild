"""Deterministic full-stream evaluation with each next-token target counted once."""

import torch
from torch.nn import functional as F

from rebuild_gpt.gpt import GPTLanguageModel


def full_corpus_loss(
    model: GPTLanguageModel, tokens: torch.Tensor, *, block_size: int
) -> float:
    """Score every adjacent token pair in nonoverlapping causal context chunks.

    Each target is counted exactly once, and the context restarts at each
    block boundary. This is not an overlapping sliding-window evaluation.
    """
    if tokens.ndim != 1 or tokens.dtype != torch.long or tokens.numel() < 2:
        raise ValueError("tokens must be a flat long stream of at least two IDs")
    if type(block_size) is not int or block_size < 1 or block_size > model.config.block_size:
        raise ValueError("block_size must be positive and at most model.config.block_size")

    device = next(model.parameters()).device
    was_training = model.training
    total_loss = 0.0
    try:
        model.eval()
        with torch.no_grad():
            for start in range(0, len(tokens) - 1, block_size):
                stop = min(start + block_size, len(tokens) - 1)
                inputs = tokens[start:stop].unsqueeze(0).to(device)
                targets = tokens[start + 1:stop + 1].unsqueeze(0).to(device)
                logits = model(inputs)
                total_loss += F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1), reduction="sum",
                ).item()
        return total_loss / (len(tokens) - 1)
    finally:
        model.train(was_training)


def full_train_val_loss(
    model: GPTLanguageModel, train: torch.Tensor, validation: torch.Tensor,
    *, block_size: int,
) -> dict[str, float]:
    """Evaluate every transition in each contiguous split, without sampling."""
    return {
        "train_loss": full_corpus_loss(model, train, block_size=block_size),
        "validation_loss": full_corpus_loss(model, validation, block_size=block_size),
    }
