"""Lesson 20: identical tokens at different positions can carry different signals."""

import torch

from rebuild_gpt.embeddings import PositionEmbedding, TokenEmbedding

torch.manual_seed(0)
token_ids = torch.tensor([[1, 1, 1], [1, 1, 1]])
tokens = TokenEmbedding(vocab_size=4, n_embd=5)(token_ids)
positions = PositionEmbedding(block_size=8, n_embd=5)(token_ids.size(1))
combined = tokens + positions

print("token vectors [B,T,C]:", tuple(tokens.shape))
print("position vectors [T,C]:", tuple(positions.shape))
print("combined vectors [B,T,C]:", tuple(combined.shape))
print("same token vector at all positions:", torch.equal(tokens[0, 0], tokens[0, 1]))
print("position-aware vectors differ:", not torch.equal(combined[0, 0], combined[0, 1]))
