"""Lesson 42: train one pre-norm transformer block to use earlier context."""

import torch

from rebuild_gpt.training import train_step
from rebuild_gpt.transformer_model import TransformerLanguageModel

torch.manual_seed(7)
model = TransformerLanguageModel(vocab_size=3, block_size=2, n_embd=8, n_head=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)

# Last token and absolute position match. Only the earlier token differs.
inputs = torch.tensor([[0, 1]] * 8 + [[2, 1]] * 8)
targets = torch.tensor([[1, 0]] * 8 + [[1, 2]] * 8)

print("loss before:", round(model.loss(inputs, targets).item(), 4))
for _ in range(110):
    train_step(model, optimizer, inputs, targets)

model.eval()
with torch.no_grad():
    print("loss after: ", round(model.loss(inputs, targets).item(), 4))
    contexts = torch.tensor([[0, 1], [2, 1]])
    print("contexts:   ", contexts.tolist())
    print("predictions:", model(contexts).argmax(dim=-1).tolist())
print("This is one transformer block, not a full multi-layer GPT.")
