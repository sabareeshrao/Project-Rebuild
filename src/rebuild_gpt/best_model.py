"""Remember weights from the lowest observed validation loss."""

import math

import torch
from torch import nn


class BestValidationModel:
    """Keep an independent CPU snapshot of the best measured model state."""

    def __init__(self) -> None:
        self.best_loss = math.inf
        self.best_step: int | None = None
        self._state: dict[str, torch.Tensor] | None = None

    def consider(self, model: nn.Module, step: int, validation_loss: float) -> bool:
        """Replace the snapshot only for a strictly lower, finite loss."""
        if type(step) is not int or step < 0:
            raise ValueError("step must be a nonnegative integer")
        if (
            isinstance(validation_loss, bool)
            or not isinstance(validation_loss, (float, int))
            or not math.isfinite(validation_loss)
            or validation_loss < 0
        ):
            raise ValueError("validation_loss must be a finite nonnegative number")
        if validation_loss >= self.best_loss:
            return False
        self.best_loss = float(validation_loss)
        self.best_step = step
        self._state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }
        return True

    def restore(self, model: nn.Module) -> None:
        """Copy the best checkpoint into the same model, preserving weight tying."""
        if self._state is None:
            raise ValueError("No best model has been recorded")
        model.load_state_dict(self._state, strict=True)
