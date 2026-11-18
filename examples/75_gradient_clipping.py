"""Lesson 78: cap a large gradient before updating model weights."""

import torch
from torch import nn

from rebuild_gpt.training import train_step


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([10.]))

    def loss(self, inputs, targets):
        return (self.weight * inputs.float()).square().mean()


for threshold in (None, 0.1):
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    train_step(model, optimizer, torch.ones(1, 1), torch.zeros(1, 1),
               max_grad_norm=threshold)
    print(f"clip={threshold}: gradient={model.weight.grad.item():.2f}, "
          f"new weight={model.weight.item():.2f}")
