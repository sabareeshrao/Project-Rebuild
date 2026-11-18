"""Lesson 50: one linear layer creates Q, K and V for every attention head."""

import torch
from rebuild_gpt.qkv import FusedQKVProjection

torch.manual_seed(2)
packed = FusedQKVProjection(n_embd=8, n_head=2)
input_vectors = torch.randn(3, 5, 8)
q, k, v = packed(input_vectors)

print("input [B,T,C]:", tuple(input_vectors.shape))
print("packed weight [3*C,C]:", tuple(packed.c_attn.weight.shape))
print("Q [B,H,T,D]:", tuple(q.shape))
print("K [B,H,T,D]:", tuple(k.shape))
print("V [B,H,T,D]:", tuple(v.shape))
