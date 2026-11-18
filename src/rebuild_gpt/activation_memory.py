"""Reason about the shape of explicit attention matrices.

The estimates here describe mathematical tensor sizes, not measured
allocator peaks. Optimized SDPA kernels may avoid materializing them.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AttentionMatrixEstimate:
    batch_size: int
    n_head: int
    context_length: int
    score_elements: int
    bytes_per_matrix: int

    @property
    def score_shape(self) -> tuple[int, int, int, int]:
        return (self.batch_size, self.n_head, self.context_length,
                self.context_length)


def estimate_attention_matrix(
    *, batch_size: int, n_head: int, context_length: int,
    dtype: torch.dtype = torch.float32,
) -> AttentionMatrixEstimate:
    """Compute the storage size of one explicit [B,H,T,T] score matrix."""
    for name, size in (
        ("batch_size", batch_size), ("n_head", n_head),
        ("context_length", context_length),
    ):
        if type(size) is not int or size < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise ValueError("dtype must be a floating-point torch.dtype")
    elements = batch_size * n_head * context_length * context_length
    element_size = torch.empty((), dtype=dtype).element_size()
    return AttentionMatrixEstimate(
        batch_size, n_head, context_length,
        score_elements=elements,
        bytes_per_matrix=elements * element_size,
    )
