"""Lesson 60: use upstream nanoGPT's inference and training output contract."""

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel

torch.manual_seed(0)
model = GPTLanguageModel(GPTConfig(7, 5, 8, 2, 2, attention_impl="fused"))
inputs = torch.tensor([[0, 1, 2, 3]])
targets = torch.tensor([[1, 2, 3, -1]])

all_logits = model(inputs)
last_logits, no_loss = model.forward_nanogpt(inputs)
training_logits, training_loss = model.forward_nanogpt(inputs, targets)

print("educational full logits shape:", tuple(all_logits.shape))
print("upstream inference logits shape:", tuple(last_logits.shape))
print("upstream training logits shape:", tuple(training_logits.shape))
print("ignored target at final position; loss:", training_loss.item())
