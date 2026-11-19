"""Deterministic *training*-only batches for optimizer trajectory comparisons."""

from pathlib import Path

import torch

from rebuild_gpt.data import load_corpus, sample_batch, split_train_val
from rebuild_gpt.tokenizer import CharTokenizer


def training_parity_batches(
    corpus_path: str | Path,
    tokenizer: CharTokenizer,
    *,
    steps: int = 3,
    batch_size: int = 2,
    block_size: int = 4,
    seed: int = 29,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Create independent consecutive training windows from a private RNG.

    Each pair is [B,T] of torch.long tokens. The training portion only is
    sampled; held-out validation tokens are never used for optimizer steps.
    """
    for name, value in (("steps", steps), ("batch_size", batch_size),
                        ("block_size", block_size)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    tokens, corpus_tokenizer = load_corpus(corpus_path)
    if tokenizer.vocab != corpus_tokenizer.vocab:
        raise ValueError("Saved tokenizer vocabulary does not match corpus")
    train, _ = split_train_val(tokens, val_fraction=0.2)
    if len(train) <= block_size:
        raise ValueError("Training split is too short for block_size")
    generator = torch.Generator().manual_seed(seed)
    return tuple(
        sample_batch(train, batch_size=batch_size, block_size=block_size,
                     generator=generator)
        for _ in range(steps)
    )
