"""Lesson 28: value vectors carry the content that attention will combine."""

import torch

from rebuild_gpt.attention import ValueProjection

torch.manual_seed(10)
hidden = torch.randn(2, 4, 6)
values = ValueProjection(n_embd=6, head_size=3)(hidden)

print("hidden [B,T,C]:", tuple(hidden.shape))
print("values [B,T,D]:", tuple(values.shape))
print("Value vectors carry information; Q/K determine mixing weights.")
