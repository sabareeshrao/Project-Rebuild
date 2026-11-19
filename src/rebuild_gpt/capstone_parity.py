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


def compare_capstone_forward(
    checkpoint_path: str | Path,
    corpus_path: str | Path,
    upstream_model_path: str | Path,
    *,
    batch_size: int = 2,
    seed: int = 29,
) -> dict[str, int | float]:
    """Compare saved, possibly trained GPT weights with trusted upstream code.

    The explicit upstream Python file is imported and EXECUTED. Only use
    locally reviewed source; never silently fetch a reference from the web.
    """
    from rebuild_gpt.text_checkpoint import load_text_checkpoint
    from rebuild_gpt.upstream_parity import compare_against_upstream, load_upstream_model

    ours, tokenizer = load_text_checkpoint(checkpoint_path)
    inputs, targets = prepare_validation_parity_batch(
        corpus_path, tokenizer, ours.config, batch_size=batch_size, seed=seed,
    )
    upstream = load_upstream_model(upstream_model_path, ours.config)
    result = compare_against_upstream(ours, upstream, inputs, targets)
    return {
        "batch_size": inputs.size(0),
        "context_length": inputs.size(1),
        "matched_tensors": result.matched_tensors,
        "max_logit_error": result.max_logit_error,
        "loss_error": result.loss_error,
    }
