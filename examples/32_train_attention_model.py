"""Lesson 32: learn different next tokens from identical current tokens."""

import torch

from rebuild_gpt.single_head_model import SingleHeadLanguageModel
from rebuild_gpt.training import train_step

torch.manual_seed(7)
model = SingleHeadLanguageModel(vocab_size=3, block_size=2, n_embd=8, head_size=4)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.04)

# Both sets have the same last input token (1), but different targets.
inputs = torch.tensor([[0, 1]] * 8 + [[2, 1]] * 8)
targets = torch.tensor([[1, 0]] * 8 + [[1, 2]] * 8)

print("loss before:", round(model.loss(inputs, targets).item(), 4))
for _ in range(140):
    train_step(model, optimizer, inputs, targets)
print("loss after:", round(model.loss(inputs, targets).item(), 4))

predicted = model(torch.tensor([[0, 1], [2, 1]])).argmax(dim=-1)
print("input contexts:", [[0, 1], [2, 1]])
print("predicted next-token targets:", predicted.tolist())
print("Contrast: a bigram or position-only model cannot fit both answers at position 1.")
