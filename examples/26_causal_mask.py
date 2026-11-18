"""Lesson 26: future tokens cannot contribute to earlier positions."""

import torch

from rebuild_gpt.attention import causal_mask_scores
from rebuild_gpt.probability import stable_softmax

scores = torch.tensor([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]])
masked = causal_mask_scores(scores)
weights = stable_softmax(masked)

print("raw scores:\n", scores[0])
print("masked scores:\n", masked[0])
print("future weights are zero:\n", weights[0])
