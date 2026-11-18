"""Small, explicit training operations for an educational language model."""

import torch

from rebuild_gpt.bigram import BigramLanguageModel


def train_step(
    model: BigramLanguageModel,
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
