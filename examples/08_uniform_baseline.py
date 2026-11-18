"""Lesson 08: every possible next token receives an equal score."""

from rebuild_gpt.baselines import uniform_logits
from rebuild_gpt.tensors import token_batch

inputs = token_batch([[1, 0, 2], [2, 1, 0]])
logits = uniform_logits(inputs, vocab_size=4)
print("input shape:", tuple(inputs.shape))  # (2, 3)
print("logit shape:", tuple(logits.shape))  # (2, 3, 4)
print("first position scores:", logits[0, 0].tolist())
