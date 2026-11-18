"""Lesson 23: a matrix of query-key similarities from simple 2D vectors."""

import torch

from rebuild_gpt.attention import pairwise_dot_scores

queries = torch.tensor([[[1., 0.], [0., 1.]]])
keys = torch.tensor([[[1., 0.], [1., 1.], [0., 1.]]])

print("queries:", tuple(queries.shape))  # [B=1, T=2, D=2]
print("keys:", tuple(keys.shape))        # [B=1, S=3, D=2]
print("scores:", pairwise_dot_scores(queries, keys).tolist())
# [[[1, 1, 0], [0, 1, 1]]]
