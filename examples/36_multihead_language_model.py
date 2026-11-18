"""Lesson 36: multi-head attention makes language-model outputs context-aware."""

import torch

from rebuild_gpt.multihead_model import MultiHeadLanguageModel

torch.manual_seed(2)
model = MultiHeadLanguageModel(vocab_size=5, block_size=4, n_embd=8, n_head=2)
inputs = torch.tensor([[0, 1, 2], [3, 1, 4]])
targets = torch.tensor([[1, 2, 3], [1, 4, 0]])
logits = model(inputs)
loss = model.loss(inputs, targets)
loss.backward()

print("inputs [B,T]:", tuple(inputs.shape))
print("multi-head logits [B,T,V]:", tuple(logits.shape))
print("loss:", round(loss.item(), 4))
print("number of attention heads:", model.attention.n_head)
print("all parameter gradients available:", all(p.grad is not None for p in model.parameters()))
