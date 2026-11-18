"""Lesson 01: token IDs become a [batch, time] tensor."""

from rebuild_gpt.tensors import token_batch

batch = token_batch([[10, 20, 30, 40], [50, 60, 70, 80]])
print("tokens:\n", batch)
print("shape:", tuple(batch.shape))  # (2 sequences, 4 positions)
print("dtype:", batch.dtype)  # int64 is required for embedding lookups
