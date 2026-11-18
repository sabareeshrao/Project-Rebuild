"""Small, explicit training operations for an educational language model."""

from typing import Protocol

import torch


class TrainableLanguageModel(Protocol):
    """The two behaviors our optimizer helper needs from any model."""

    def train(self, mode: bool = True) -> object: ...

    def loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor: ...


def train_step(
    model: TrainableLanguageModel,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """Apply one gradient update and return the loss before that update."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = model.loss(inputs, targets)
    loss.backward()
    optimizer.step()
    return loss.detach().item()
