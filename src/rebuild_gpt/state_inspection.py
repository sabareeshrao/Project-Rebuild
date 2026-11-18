"""Inspect saved tensor names, shapes, and shared learnable weights."""

from torch import nn


def state_shapes(model: nn.Module) -> dict[str, tuple[int, ...]]:
    """List the shapes of the named tensors in a model's state_dict."""
    return {name: tuple(tensor.shape) for name, tensor in model.state_dict().items()}


def shared_parameters(model: nn.Module) -> list[tuple[str, ...]]:
    """Find names referring to the very same Parameter, not just equal values."""
    by_identity: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        by_identity.setdefault(id(parameter), []).append(name)
    return [tuple(names) for names in by_identity.values() if len(names) > 1]
