"""Repeatable held-out loss estimates for our educational GPT trainer."""

import torch

from rebuild_gpt.data import sample_batch
from rebuild_gpt.gpt import GPTLanguageModel


def estimate_split_loss(
    model: GPTLanguageModel,
    tokens: torch.Tensor,
    *,
    batch_size: int,
    block_size: int,
    eval_batches: int = 4,
    seed: int = 0,
) -> float:
    """Average fixed, seeded batches without altering training mode or RNG state."""
    if type(eval_batches) is not int or eval_batches < 1:
        raise ValueError("eval_batches must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if type(block_size) is not int or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if block_size > model.config.block_size:
        raise ValueError("block_size exceeds the model context window")

    generator = torch.Generator(device=tokens.device).manual_seed(seed)
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            losses = []
            for _ in range(eval_batches):
                inputs, targets = sample_batch(
                    tokens, batch_size=batch_size, block_size=block_size,
                    generator=generator,
                )
                losses.append(model.loss(inputs, targets).item())
        return sum(losses) / len(losses)
    finally:
        model.train(was_training)


def estimate_train_val_loss(
    model: GPTLanguageModel,
    train_tokens: torch.Tensor,
    val_tokens: torch.Tensor,
    *,
    batch_size: int,
    block_size: int,
    eval_batches: int = 4,
    seed: int = 0,
) -> dict[str, float]:
    """Compare both splits using independent, repeatable batch streams."""
    return {
        "train_loss": estimate_split_loss(
            model, train_tokens, batch_size=batch_size, block_size=block_size,
            eval_batches=eval_batches, seed=seed,
        ),
        "validation_loss": estimate_split_loss(
            model, val_tokens, batch_size=batch_size, block_size=block_size,
            eval_batches=eval_batches, seed=seed + 1,
        ),
    }
