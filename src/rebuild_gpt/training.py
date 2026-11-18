"""Small, explicit training operations for an educational language model."""

import math
from typing import Protocol

import torch


class TrainableLanguageModel(Protocol):
    """The behaviors our optimizer helper needs from any model."""

    def train(self, mode: bool = True) -> object: ...

    def loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor: ...

    def parameters(self): ...


def train_step(
    model: TrainableLanguageModel,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    max_grad_norm: float | None = None,
) -> float:
    """Backpropagate, optionally clip global gradient norm, then update weights."""
    if max_grad_norm is not None and (
        isinstance(max_grad_norm, bool)
        or not isinstance(max_grad_norm, (float, int))
        or not math.isfinite(max_grad_norm)
        or max_grad_norm <= 0
    ):
        raise ValueError("max_grad_norm must be finite and positive or None")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = model.loss(inputs, targets)
    loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_grad_norm, error_if_nonfinite=True
        )
    optimizer.step()
    return loss.detach().item()
