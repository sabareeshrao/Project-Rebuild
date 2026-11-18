"""Lesson 21: add token and position vectors and project into next-token logits."""

import torch

from rebuild_gpt.position_model import EmbeddingLanguageModel

torch.manual_seed(0)
model = EmbeddingLanguageModel(vocab_size=5, block_size=4, n_embd=6)
inputs = torch.tensor([[1, 2, 3], [0, 2, 1]])
targets = torch.tensor([[2, 3, 4], [2, 1, 0]])
logits = model(inputs)
loss = model.loss(inputs, targets)
loss.backward()

print("logits [B,T,V]:", tuple(logits.shape))
print("cross-entropy:", round(loss.item(), 4))
print("token embedding has gradient:", model.tokens.lookup.weight.grad is not None)
print("position embedding has gradient:", model.positions.lookup.weight.grad is not None)
print("projection head has gradient:", model.head.linear.weight.grad is not None)
