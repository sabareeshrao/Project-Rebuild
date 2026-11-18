"""Lesson 37: demonstrate that multi-head attention learns from earlier context."""

import torch

from rebuild_gpt.multihead_model import MultiHeadLanguageModel
from rebuild_gpt.training import train_step

torch.manual_seed(7)
model = MultiHeadLanguageModel(vocab_size=3, block_size=2, n_embd=8, n_head=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.04)

# The final input token is always 1, but its desired next token depends
# on whether the previous input was 0 or 2.
inputs = torch.tensor([[0, 1]] * 8 + [[2, 1]] * 8)
targets = torch.tensor([[1, 0]] * 8 + [[1, 2]] * 8)

print("loss before:", round(model.loss(inputs, targets).item(), 4))
for _ in range(120):
    train_step(model, optimizer, inputs, targets)

model.eval()
with torch.no_grad():
    print("loss after: ", round(model.loss(inputs, targets).item(), 4))
    contexts = torch.tensor([[0, 1], [2, 1]])
    print("contexts:   ", contexts.tolist())
    print("predictions:", model(contexts).argmax(dim=-1).tolist())
print("This proves past-token conditioning, not that multiple heads are always better.")
