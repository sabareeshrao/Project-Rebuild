"""Lesson 25: scaling limits the growth of attention scores with head width."""

import torch

from rebuild_gpt.attention import pairwise_dot_scores, scaled_dot_scores
from rebuild_gpt.probability import stable_softmax

queries = torch.tensor([[[2., 0.]]])
keys = torch.tensor([[[1., 0.], [2., 1.]]])

raw = pairwise_dot_scores(queries, keys)
scaled = scaled_dot_scores(queries, keys)

print("unscaled dot scores:", raw.tolist())
print("scaled dot scores:", scaled.tolist())
print("unscaled probabilities:", stable_softmax(raw).tolist())
print("scaled probabilities:", stable_softmax(scaled).tolist())
