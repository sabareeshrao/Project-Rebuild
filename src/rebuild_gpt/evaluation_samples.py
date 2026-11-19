"""Inspect the per-batch losses underlying all three sampled evaluators."""

import torch

from rebuild_gpt.data import sample_batch
from rebuild_gpt.eval_cache import PreparedEvaluationBatches
from rebuild_gpt.eval_index_plan import EvaluationIndexPlan
from rebuild_gpt.gpt import GPTLanguageModel


def collect_split_losses(
    model: GPTLanguageModel,
    source: torch.Tensor | PreparedEvaluationBatches | EvaluationIndexPlan,
    *,
    batch_size: int | None = None,
    block_size: int | None = None,
    eval_batches: int = 4,
    seed: int = 0,
) -> tuple[float, ...]:
    """Evaluate each batch once, preserving model mode and global RNG.

    Raw-token sources regenerate seeded batches. Prepared/index plans
    reuse their original seeds, shapes and examples. Neither model
    weights nor the optimizer are modified.
    """
    if isinstance(source, PreparedEvaluationBatches):
        if not source.batches:
            raise ValueError("source must contain nonempty batches")
        context = source.block_size
        batches = source.batches
    elif isinstance(source, EvaluationIndexPlan):
        if not source.starts:
            raise ValueError("source must contain nonempty starts")
        context = source.block_size
        batches = source.batches()
    elif isinstance(source, torch.Tensor):
        for name, value in (
            ("batch_size", batch_size), ("block_size", block_size),
            ("eval_batches", eval_batches),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        context = block_size
        cpu_tokens = source.cpu()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        def regenerated():
            for _ in range(eval_batches):
                yield sample_batch(
                    cpu_tokens, batch_size=batch_size,
                    block_size=block_size, generator=generator,
                )
        batches = regenerated()
    else:
        raise ValueError("source must be token IDs or a prepared evaluation plan")
    if context > model.config.block_size:
        raise ValueError("block_size exceeds the model context window")

    device = next(model.parameters()).device
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            return tuple(
                model.loss(inputs.to(device), targets.to(device)).item()
                for inputs, targets in batches
            )
    finally:
        model.train(was_training)
