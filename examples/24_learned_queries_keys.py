"""Lesson 24: hidden vectors become trainable queries and keys."""

import torch

from rebuild_gpt.attention import QueryKeyProjection, pairwise_dot_scores

torch.manual_seed(3)
hidden = torch.randn(2, 4, 6)  # two sequences, four tokens, six features
projector = QueryKeyProjection(n_embd=6, head_size=3)
queries, keys = projector(hidden)

print("input [B,T,C]:", tuple(hidden.shape))
print("queries [B,T,D]:", tuple(queries.shape))
print("keys [B,T,D]:", tuple(keys.shape))
print("raw scores [B,T,T]:", tuple(pairwise_dot_scores(queries, keys).shape))
