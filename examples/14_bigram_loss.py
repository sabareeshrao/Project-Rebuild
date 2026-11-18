"""Lesson 14: the model's wrong guesses produce a training signal."""

import torch

from rebuild_gpt.bigram import BigramLanguageModel

torch.manual_seed(0)
model = BigramLanguageModel(vocab_size=3)
inputs = torch.tensor([[0, 1, 2]])
targets = torch.tensor([[1, 2, 0]])

loss = model.loss(inputs, targets)
loss.backward()
print("initial loss:", round(loss.item(), 4))
print("gradient shape:", tuple(model.token_to_logits.weight.grad.shape))
print("gradient magnitude:", model.token_to_logits.weight.grad.abs().sum().item())
