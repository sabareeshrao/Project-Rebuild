"""Lesson 02: each input token predicts the token immediately after it."""

from rebuild_gpt.tensors import next_token_pairs, token_batch

batch = token_batch([[10, 20, 30, 40]])
inputs, targets = next_token_pairs(batch)
print("inputs:", inputs.tolist())   # [[10, 20, 30]]
print("targets:", targets.tolist())  # [[20, 30, 40]]
