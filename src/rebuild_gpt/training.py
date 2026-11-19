"""Small, explicit training operations for an educational language model."""

import math
from typing import Protocol

import torch

from rebuild_gpt.precision import autocast_context, resolve_precision


class TrainableLanguageModel(Protocol):
    """The behaviors our optimizer helper needs from any model."""

    def train(self, mode: bool = True) -> object: ...

    def loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor: ...

    def parameters(self): ...

def _finish_optimizer_step(
    model: TrainableLanguageModel,
    optimizer: torch.optim.Optimizer,
    max_grad_norm: float | None,
    scaler: torch.cuda.amp.GradScaler | None,
) -> None:
    """Unscale BEFORE clipping; a skipped FP16 step must not update weights."""
    if scaler is not None and max_grad_norm is not None:
        scaler.unscale_(optimizer)
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            # FP16 GradScaler detects overflow and skips the update.
            # Do not raise before the scaler can handle nonfinite grads.
            model.parameters(), max_grad_norm,
            error_if_nonfinite=(scaler is None)
        )
    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()


def _check_amp(
    model: TrainableLanguageModel, precision: str,
    scaler: torch.cuda.amp.GradScaler | None,
) -> torch.device:
    device = next(iter(model.parameters())).device
    resolve_precision(device, precision)
    if precision == "fp16" and scaler is None:
        raise ValueError("fp16 training requires a gradient scaler")
    if precision != "fp16" and scaler is not None:
        raise ValueError("gradient scaler should only be used for fp16")
    return device


def train_step(
    model: TrainableLanguageModel,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    max_grad_norm: float | None = None,
    precision: str = "fp32",
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> float:
    """Autocast forward, backpropagate, unscale, clip, and update weights."""
    if max_grad_norm is not None and (
        isinstance(max_grad_norm, bool)
        or not isinstance(max_grad_norm, (float, int))
        or not math.isfinite(max_grad_norm)
        or max_grad_norm <= 0
    ):
        raise ValueError("max_grad_norm must be finite and positive or None")
    device = _check_amp(model, precision, scaler)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, precision):
        loss = model.loss(inputs, targets)
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()
    _finish_optimizer_step(model, optimizer, max_grad_norm, scaler)
    return loss.detach().item()


def train_accumulated_step(
    model: TrainableLanguageModel,
    optimizer: torch.optim.Optimizer,
    microbatches: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    max_grad_norm: float | None = None,
    precision: str = "fp32",
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> float:
    """Average per-target losses across microbatches; take one update.

    Requires model.loss to be a mean over all target elements. An uneven
    final microbatch receives weight proportional to its target count.
    """
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
    same_size = True
    total_targets = 0
    for inputs, targets in microbatches:
        if (
            inputs.ndim < 1 or targets.ndim < 1 or inputs.numel() == 0
            or targets.numel() == 0
            or inputs.shape[1:] != input_shape[1:]
            or targets.shape[1:] != target_shape[1:]
            or inputs.size(0) != targets.size(0)
        ):
            raise ValueError("microbatches must have matching nonempty batch shapes")
        same_size &= inputs.shape == input_shape and targets.shape == target_shape
        total_targets += targets.numel()

    device = _check_amp(model, precision, scaler)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for inputs, targets in microbatches:
        with autocast_context(device, precision):
            loss = model.loss(inputs, targets)
        # The loss is a mean over target elements. For an uneven final
        # microbatch, weight by its token count, not one per chunk.
        # Retain the previous arithmetic for equal-sized chunks.
        weight = targets.numel() / total_targets
        normalized = (
            loss / len(microbatches) if same_size else loss * weight
        )
        if scaler is None:
            normalized.backward()
        else:
            scaler.scale(normalized).backward()
        total_loss += (
            loss.detach().item() if same_size
            else loss.detach().item() * weight
        )
    _finish_optimizer_step(model, optimizer, max_grad_norm, scaler)
    return total_loss / len(microbatches) if same_size else total_loss
