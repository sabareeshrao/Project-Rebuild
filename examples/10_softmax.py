"""Lesson 10: convert scores (logits) into a next-token distribution."""

import torch

from rebuild_gpt.probability import stable_softmax

logits = torch.tensor([2.0, 1.0, 0.0])
probs = stable_softmax(logits)
print("logits:", logits.tolist())
print("probabilities:", probs.tolist())
print("sum:", probs.sum().item())

large_logits = torch.tensor([1000.0, 1001.0])
print("stable for large logits:", stable_softmax(large_logits).tolist())
