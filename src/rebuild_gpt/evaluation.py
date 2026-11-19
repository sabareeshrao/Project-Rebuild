"""Repeatable held-out loss estimates for our educational GPT trainer."""

import torch

from rebuild_gpt.data import sample_batch
from rebuild_gpt.eval_cache import PreparedEvaluationBatches
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

    # Keep the sampling stream on CPU, even when the model is on CUDA.
    # This avoids tying sampled indices to the device's RNG implementation.
    cpu_tokens = tokens.cpu()
    device = next(model.parameters()).device
    generator = torch.Generator(device="cpu").manual_seed(seed)
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            losses = []
            for _ in range(eval_batches):
                inputs, targets = sample_batch(
                    cpu_tokens, batch_size=batch_size, block_size=block_size,
                    generator=generator,
                )
                losses.append(model.loss(inputs.to(device), targets.to(device)).item())
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


def evaluate_prepared_split_loss(
    model: GPTLanguageModel,
    prepared: PreparedEvaluationBatches,
) -> float:
    """Evaluate fixed CPU batches while preserving mode and RNG.

    Only batch generation is cached; device transfers and model forward
    passes still occur on every call to reflect current model weights.
    """
    if not isinstance(prepared, PreparedEvaluationBatches) or not prepared.batches:
        raise ValueError("prepared must contain nonempty evaluation batches")
    if prepared.block_size > model.config.block_size:
        raise ValueError("block_size exceeds the model context window")
    device = next(model.parameters()).device
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            losses = [
                model.loss(inputs.to(device), targets.to(device)).item()
                for inputs, targets in prepared.batches
            ]
        return sum(losses) / len(losses)
    finally:
        model.train(was_training)


def evaluate_prepared_train_val_loss(
    model: GPTLanguageModel,
    train: PreparedEvaluationBatches,
    validation: PreparedEvaluationBatches,
) -> dict[str, float]:
    """Evaluate train/validation using independent cached batch plans."""
    return {
        "train_loss": evaluate_prepared_split_loss(model, train),
        "validation_loss": evaluate_prepared_split_loss(model, validation),
    }
