"""Lesson 15: repeated optimizer steps improve a simple token pattern."""

import torch

from rebuild_gpt.bigram import BigramLanguageModel
from rebuild_gpt.training import train_step

torch.manual_seed(2)
model = BigramLanguageModel(vocab_size=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
inputs = torch.tensor([[0, 1, 0, 1]])
targets = torch.tensor([[1, 0, 1, 0]])

print("before:", round(model.loss(inputs, targets).item(), 4))
for _ in range(40):
    train_step(model, optimizer, inputs, targets)
print("after:", round(model.loss(inputs, targets).item(), 4))
