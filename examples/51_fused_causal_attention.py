"""Lesson 51: compute all attention heads in one batch with a causal mask."""

import torch
from rebuild_gpt.fused_attention import FusedCausalSelfAttention

torch.manual_seed(7)
layer = FusedCausalSelfAttention(n_embd=8, n_head=2)
hidden = torch.randn(2, 5, 8)
output = layer(hidden)

print("input [B,T,C]:", tuple(hidden.shape))
print("QKV weight [3C,C]:", tuple(layer.qkv.c_attn.weight.shape))
print("output [B,T,C]:", tuple(output.shape))
print("fused attention has probability and residual dropout")
