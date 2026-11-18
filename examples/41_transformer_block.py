"""Lesson 41: one complete pre-norm GPT-style block."""

import torch

from rebuild_gpt.block import TransformerBlock

torch.manual_seed(1)
block = TransformerBlock(n_embd=8, n_head=2)
hidden = torch.randn(2, 4, 8, requires_grad=True)
result = block(hidden)

print("input [B,T,C]:", tuple(hidden.shape))
print("output [B,T,C]:", tuple(result.shape))
print("has attention:", type(block.attention).__name__)
print("has MLP:", type(block.mlp).__name__)
result.mean().backward()
print("attention and MLP can receive gradients:",
      all(param.grad is not None for param in block.parameters()))
