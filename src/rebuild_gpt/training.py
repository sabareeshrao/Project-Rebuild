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


def train_accumulated_step(
    model: TrainableLanguageModel,
    optimizer: torch.optim.Optimizer,
    microbatches: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    max_grad_norm: float | None = None,
) -> float:
    """Average equally sized microbatch losses, then take exactly one update."""
    if not microbatches:
        raise ValueError("microbatches must contain at least one batch")
    if max_grad_norm is not None and (
        isinstance(max_grad_norm, bool)
        or not isinstance(max_grad_norm, (int, float))
        or not math.isfinite(max_grad_norm)
        or max_grad_norm <= 0
    ):
        raise ValueError("max_grad_norm must be finite and positive or None")
    input_shape = microbatches[0][0].shape
    target_shape = microbatches[0][1].shape
    for inputs, targets in microbatches:
        if (
            inputs.ndim < 1 or targets.ndim < 1 or inputs.numel() == 0
            or inputs.shape != input_shape or targets.shape != target_shape
            or inputs.size(0) != targets.size(0)
        ):
            raise ValueError("microbatches must have matching nonempty batch shapes")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for inputs, targets in microbatches:
        loss = model.loss(inputs, targets)
        (loss / len(microbatches)).backward()
        total_loss += loss.detach().item()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_grad_norm, error_if_nonfinite=True
        )
    optimizer.step()
    return total_loss / len(microbatches)
