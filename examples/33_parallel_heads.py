"""Lesson 33: H attention heads learn different representations in parallel."""

import torch

from rebuild_gpt.multihead import ParallelAttentionHeads

torch.manual_seed(6)
hidden = torch.randn(2, 4, 8)  # batch=2, time=4, model width=8
heads = ParallelAttentionHeads(n_embd=8, n_head=3, head_size=2)
separate = [head(hidden) for head in heads.heads]
combined = heads(hidden)

print("input [B,T,C]:", tuple(hidden.shape))
print("each head [B,T,D]:", tuple(separate[0].shape))
print("concatenated [B,T,H*D]:", tuple(combined.shape))
print("independent head modules:", len(heads.heads))
