"""Lesson 34: H heads concatenate, then a trainable projection mixes features."""

import torch

from rebuild_gpt.multihead import MultiHeadAttention

torch.manual_seed(9)
layer = MultiHeadAttention(n_embd=8, n_head=2)
hidden = torch.randn(2, 4, 8)

print("hidden [B,T,C]:", tuple(hidden.shape))
print("number of heads:", layer.n_head)
print("features per head:", layer.head_size)
print("concatenated [B,T,C]:", tuple(layer.heads(hidden).shape))
print("projected [B,T,C]:", tuple(layer(hidden).shape))
