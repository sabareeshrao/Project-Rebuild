"""Lesson 19: project each embedding vector into next-token prediction scores."""

import torch

from rebuild_gpt.embeddings import TokenEmbedding, VocabularyProjection

torch.manual_seed(0)
tokens = torch.tensor([[0, 1, 2], [2, 1, 0]])
emb = TokenEmbedding(vocab_size=4, n_embd=6)
head = VocabularyProjection(n_embd=6, vocab_size=4)

vectors = emb(tokens)
logits = head(vectors)
print("vectors [B,T,C]:", tuple(vectors.shape))
print("logits  [B,T,V]:", tuple(logits.shape))
print("projection parameters:", sum(p.numel() for p in head.parameters()))
