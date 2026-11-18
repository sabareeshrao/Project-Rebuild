"""Lesson 11: confidence in the correct token reduces prediction loss."""

import torch

from rebuild_gpt.probability import next_token_cross_entropy

target = torch.tensor([[1]])
wrong = torch.tensor([[[3.0, 0.0, 0.0]]])
right = torch.tensor([[[0.0, 3.0, 0.0]]])

print("loss when incorrect token is favored:", next_token_cross_entropy(wrong, target).item())
print("loss when correct token is favored:", next_token_cross_entropy(right, target).item())
