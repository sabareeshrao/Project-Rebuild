"""Explain model memory without requiring a GPU.

These are *estimates* for parameters, gradient tensors, and AdamW moments.
They do not include activations, kernels, allocator overhead, or workspace.
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MemoryEstimate:
    parameter_bytes: int
    buffer_bytes: int
    gradient_bytes: int
    adamw_moment_bytes: int
    trainable_parameters: int

    @property
    def estimated_training_bytes(self) -> int:
        """Model storage + full gradient buffers + two AdamW moment tensors."""
        return (
            self.parameter_bytes + self.buffer_bytes
            + self.gradient_bytes + self.adamw_moment_bytes
        )


def estimate_model_memory(model: nn.Module) -> MemoryEstimate:
    """Estimate persistent storage and typical AdamW training tensor sizes.

    nn.Module.parameters() de-duplicates a tied Parameter object. Track
    identities explicitly for clarity, including shared buffers.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    parameters = list({id(p): p for p in model.parameters()}.values())
    buffers = list({id(b): b for b in model.buffers()}.values())
    trainable = [p for p in parameters if p.requires_grad]
    size = lambda t: t.numel() * t.element_size()
    return MemoryEstimate(
        parameter_bytes=sum(map(size, parameters)),
        buffer_bytes=sum(map(size, buffers)),
        gradient_bytes=sum(map(size, trainable)),
        adamw_moment_bytes=2 * sum(map(size, trainable)),
        trainable_parameters=sum(p.numel() for p in trainable),
    )


def mebibytes(byte_count: int) -> float:
    """Convert bytes to MiB (1 MiB = 1,048,576 bytes)."""
    if type(byte_count) is not int or byte_count < 0:
        raise ValueError("byte_count must be a nonnegative integer")
    return byte_count / (1024 ** 2)
