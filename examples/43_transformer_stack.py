"""Lesson 43: chain three independent causal transformer blocks."""

import torch

from rebuild_gpt.stack import TransformerStack

torch.manual_seed(2)
stack = TransformerStack(n_embd=8, n_head=2, n_layer=3)
hidden = torch.randn(2, 4, 8, requires_grad=True)
result = stack(hidden)

print("blocks:", len(stack.blocks))
print("input [B,T,C]:", tuple(hidden.shape))
print("output [B,T,C]:", tuple(result.shape))
result.square().mean().backward()
print("all blocks receive gradients:",
      all(param.grad is not None for param in stack.parameters()))
