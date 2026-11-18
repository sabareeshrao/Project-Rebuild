"""Lesson 47: train a two-block GPT to use earlier-token context."""

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.parameters import count_parameters
from rebuild_gpt.training import train_step

torch.manual_seed(7)
config = GPTConfig(vocab_size=3, block_size=2, n_embd=8, n_head=2, n_layer=2)
model = GPTLanguageModel(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.025)

# The same current token (1), at the same absolute position (1),
# must predict different tokens depending on the preceding token.
inputs = torch.tensor([[0, 1]] * 8 + [[2, 1]] * 8)
targets = torch.tensor([[1, 0]] * 8 + [[1, 2]] * 8)

print("blocks:", config.n_layer, "parameters:", count_parameters(model))
print("loss before:", round(model.loss(inputs, targets).item(), 4))
for _ in range(120):
    train_step(model, optimizer, inputs, targets)

model.eval()
with torch.no_grad():
    print("loss after:", round(model.loss(inputs, targets).item(), 4))
    contexts = torch.tensor([[0, 1], [2, 1]])
    print("contexts:", contexts.tolist())
    print("predictions:", model(contexts).argmax(dim=-1).tolist())
print("This is a toy learning exercise, not a measure of language quality.")
