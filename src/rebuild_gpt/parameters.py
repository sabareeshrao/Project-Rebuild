"""Count independent stored parameters rather than tensor appearances."""

from torch import nn


def count_parameters(model: nn.Module, *, trainable_only: bool = True) -> int:
    """Count parameter scalars; optionally include frozen parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )
