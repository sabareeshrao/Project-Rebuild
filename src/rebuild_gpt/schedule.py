"""NanoGPT-style learning-rate warmup and cosine decay."""

import math


def cosine_warmup_lr(
    iteration: int,
    *,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    decay_steps: int,
) -> float:
    """Compute the rate at zero-based iteration, matching nanoGPT's get_lr."""
    if type(iteration) is not int or iteration < 0:
        raise ValueError("iteration must be a nonnegative integer")
    if type(warmup_steps) is not int or warmup_steps < 0:
        raise ValueError("warmup_steps must be a nonnegative integer")
    if type(decay_steps) is not int or decay_steps <= warmup_steps:
        raise ValueError("decay_steps must be greater than warmup_steps")
    if (
        isinstance(max_lr, bool) or not isinstance(max_lr, (int, float))
        or not math.isfinite(max_lr) or max_lr <= 0
    ):
        raise ValueError("max_lr must be finite and positive")
    if (
        isinstance(min_lr, bool) or not isinstance(min_lr, (int, float))
        or not math.isfinite(min_lr) or min_lr < 0 or min_lr > max_lr
    ):
        raise ValueError("min_lr must be finite and between zero and max_lr")

    if iteration < warmup_steps:
        return max_lr * (iteration + 1) / (warmup_steps + 1)
    if iteration > decay_steps:
        return float(min_lr)
    progress = (iteration - warmup_steps) / (decay_steps - warmup_steps)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr + coefficient * (max_lr - min_lr))
