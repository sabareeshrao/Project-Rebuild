"""Lesson 40: apply two linear layers and GELU to each token vector."""

import torch

from rebuild_gpt.feedforward import FeedForward

torch.manual_seed(2)
mlp = FeedForward(n_embd=6)
hidden = torch.randn(2, 4, 6)
output = mlp(hidden)

print("input shape [B,T,C]:", tuple(hidden.shape))
print("expanded width:", mlp.expand.out_features)
print("output shape [B,T,C]:", tuple(output.shape))
print("GELU module:", mlp.activation)
