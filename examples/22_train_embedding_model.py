"""Lesson 22: a position-aware model learns targets without context mixing."""

import torch

from rebuild_gpt.position_model import EmbeddingLanguageModel
from rebuild_gpt.training import train_step

torch.manual_seed(4)
model = EmbeddingLanguageModel(vocab_size=2, block_size=4, n_embd=8)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)

# Every input is token 0. Only position distinguishes the target!
inputs = torch.tensor([[0, 0, 0, 0]] * 8)
targets = torch.tensor([[0, 1, 0, 1]] * 8)

print("loss before:", round(model.loss(inputs, targets).item(), 4))
for _ in range(70):
    train_step(model, optimizer, inputs, targets)
print("loss after: ", round(model.loss(inputs, targets).item(), 4))
print("predictions: ", model(inputs).argmax(dim=-1)[0].tolist())
print("Position-aware is not yet context-aware: attention comes next.")
