"""Lesson 83: average two microbatch gradients before one optimizer step."""

import copy
import torch
from torch import nn

from rebuild_gpt.training import train_accumulated_step, train_step


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[1.0, -0.5]]))

    def loss(self, inputs, targets):
        return (inputs.float() @ self.weight.T - targets.float()).square().mean()


torch.manual_seed(7)
large = ToyModel()
small = copy.deepcopy(large)
batch_x = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [2., 0.]])
batch_y = torch.tensor([[1.], [-1.], [0.], [2.]])
large_loss = train_step(large, torch.optim.SGD(large.parameters(), lr=0.1), batch_x, batch_y)
micro = [(batch_x[:2], batch_y[:2]), (batch_x[2:], batch_y[2:])]
small_loss = train_accumulated_step(small, torch.optim.SGD(small.parameters(), lr=0.1), micro)
print("full-batch loss:", round(large_loss, 6))
print("accumulated loss:", round(small_loss, 6))
print("same updated weights:", torch.allclose(large.weight, small.weight, atol=1e-6))
