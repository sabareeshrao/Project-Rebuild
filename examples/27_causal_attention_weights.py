"""Lesson 27: normalize a learned, scaled and causally masked score matrix."""

import torch

from rebuild_gpt.attention import CausalAttentionWeights
from rebuild_gpt.embeddings import TokenEmbedding

torch.manual_seed(4)
ids = torch.tensor([[1, 2, 0, 2]])
hidden = TokenEmbedding(vocab_size=3, n_embd=6)(ids)
attention = CausalAttentionWeights(n_embd=6, head_size=3)
weights = attention(hidden)

print("input token IDs:", ids.tolist())
print("weights [B,T,T]:", tuple(weights.shape))
print("attention matrix:\n", weights[0].detach())
print("sum across each query's keys:", weights.sum(dim=-1).detach().tolist())
print("These are only weights: value aggregation comes in a later lesson.")
