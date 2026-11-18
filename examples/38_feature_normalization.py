"""Lesson 38: center and scale each token's features, independently."""

import torch

from rebuild_gpt.normalization import normalize_features

hidden = torch.tensor([[[1., 2., 3., 4.], [5., 7., 9., 11.]]])
normalized = normalize_features(hidden)
print("input shape [B,T,C]:", tuple(hidden.shape))
print("normalized:", normalized.tolist())
print("means:", normalized.mean(dim=-1).tolist())
print("variances:", normalized.var(dim=-1, unbiased=False).tolist())
