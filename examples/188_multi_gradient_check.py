"""Lesson 295: check multiple token and positional embedding gradients."""

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.gradient_coordinate_sweep import check_gradient_coordinates


if __name__ == "__main__":
    torch.manual_seed(17)
    model = GPTLanguageModel(GPTConfig(
        9, 5, 8, 2, 1, attention_impl="fused",
    )).double()
    inputs = torch.tensor([[0, 1, 2, 3]])
    targets = torch.tensor([[1, 2, 3, -1]])
    print(check_gradient_coordinates(model, inputs, targets))
