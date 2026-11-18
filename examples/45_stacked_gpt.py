"""Lesson 45: a configurable decoder-only GPT forward pass and loss."""

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel

torch.manual_seed(4)
config = GPTConfig(vocab_size=5, block_size=6, n_embd=8, n_head=2, n_layer=3)
model = GPTLanguageModel(config)

inputs = torch.tensor([[0, 1, 2, 3], [4, 3, 2, 1]])
targets = torch.tensor([[1, 2, 3, 4], [3, 2, 1, 0]])
logits = model(inputs)
loss = model.loss(inputs, targets)
loss.backward()

print("transformer depth:", len(model.blocks.blocks))
print("input [B,T]:", tuple(inputs.shape))
print("logits [B,T,V]:", tuple(logits.shape))
print("loss:", round(loss.item(), 4))
print("all parameters receive gradients:",
      all(p.grad is not None for p in model.parameters()))
