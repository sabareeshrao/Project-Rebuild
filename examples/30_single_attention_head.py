"""Lesson 30: compose Q, K and V into one causal self-attention head."""

import torch

from rebuild_gpt.attention import CausalSelfAttentionHead
from rebuild_gpt.embeddings import TokenEmbedding

torch.manual_seed(11)
ids = torch.tensor([[1, 2, 0, 2]])
hidden = TokenEmbedding(vocab_size=3, n_embd=6)(ids)
head = CausalSelfAttentionHead(n_embd=6, head_size=3)

mixed = head(hidden)
weights = head.attention_weights(hidden)

print("input embeddings [B,T,C]:", tuple(hidden.shape))
print("attention weights [B,T,T]:", tuple(weights.shape))
print("mixed values [B,T,D]:", tuple(mixed.shape))
print("future attention weights are zero:", bool((weights.triu(1) == 0).all()))
