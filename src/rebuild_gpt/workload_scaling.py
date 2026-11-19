"""Estimate workload shape and explicit attention-score storage, not GPU peaks."""

from dataclasses import dataclass

import torch

from rebuild_gpt.activation_memory import estimate_attention_matrix


@dataclass(frozen=True)
class WorkloadEstimate:
    """Quantities for one transformer training configuration."""

    batch_size: int
    context_length: int
    n_head: int
    n_layer: int
    gradient_accumulation_steps: int
    tokens_per_microbatch: int
    tokens_per_optimizer_step: int
    score_bytes_per_layer: int
    sum_score_bytes_all_layers: int


def estimate_workload(
    *, batch_size: int, context_length: int, n_head: int,
    n_layer: int, gradient_accumulation_steps: int = 1,
    dtype: torch.dtype = torch.float32,
) -> WorkloadEstimate:
    """Calculate tokens/update and one score matrix per layer.

    The sum across layers is an arithmetic comparison, *not* a memory
    peak: real autograd saves additional tensors, and SDPA may avoid
    explicit score matrices altogether.
    """
    for name, value in (
        ("n_layer", n_layer),
        ("gradient_accumulation_steps", gradient_accumulation_steps),
    ):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    matrix = estimate_attention_matrix(
        batch_size=batch_size, n_head=n_head,
        context_length=context_length, dtype=dtype,
    )
    tokens = batch_size * context_length
    return WorkloadEstimate(
        batch_size=batch_size,
        context_length=context_length,
        n_head=n_head,
        n_layer=n_layer,
        gradient_accumulation_steps=gradient_accumulation_steps,
        tokens_per_microbatch=tokens,
        tokens_per_optimizer_step=tokens * gradient_accumulation_steps,
        score_bytes_per_layer=matrix.bytes_per_matrix,
        sum_score_bytes_all_layers=matrix.bytes_per_matrix * n_layer,
    )
