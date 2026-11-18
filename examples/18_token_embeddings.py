"""Lesson 18: integer token IDs become learnable vector representations."""

import torch

from rebuild_gpt.embeddings import TokenEmbedding

torch.manual_seed(0)
embedding = TokenEmbedding(vocab_size=5, n_embd=3)
ids = torch.tensor([[0, 1, 2], [3, 1, 4]])
vectors = embedding(ids)

print("IDs shape [B,T]:", tuple(ids.shape))
print("vectors shape [B,T,C]:", tuple(vectors.shape))
print("embedding weight shape [V,C]:", tuple(embedding.lookup.weight.shape))
print("same token, same vector:", torch.equal(vectors[0, 1], vectors[1, 1]))
