"""Use held-out corpus windows for an apples-to-apples GPT parity check."""

from pathlib import Path

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.data import load_corpus, sample_batch, split_train_val
from rebuild_gpt.tokenizer import CharTokenizer


def prepare_validation_parity_batch(
    corpus_path: str | Path,
    tokenizer: CharTokenizer,
    config: GPTConfig,
    *,
    batch_size: int = 2,
    seed: int = 29,
    mask_last: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample real held-out next-token pairs with an optional ignored target."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(mask_last) is not bool:
        raise ValueError("mask_last must be boolean")

    stream, corpus_tokenizer = load_corpus(corpus_path)
    if tokenizer.vocab != corpus_tokenizer.vocab:
        raise ValueError("Saved tokenizer vocabulary does not match corpus")
    _, validation = split_train_val(stream, val_fraction=0.2)
    inputs, targets = sample_batch(
        validation,
        batch_size=batch_size,
        block_size=config.block_size,
        generator=torch.Generator().manual_seed(seed),
    )
    if mask_last:
        targets = targets.clone()
        targets[:, -1] = -1  # Upstream forward ignores -1 loss targets.
    return inputs, targets
