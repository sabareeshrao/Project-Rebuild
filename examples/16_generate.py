"""Lesson 16: use one sampled token as the next generation input."""

import torch

from rebuild_gpt.bigram import BigramLanguageModel

torch.manual_seed(0)
model = BigramLanguageModel(vocab_size=3)
start = torch.tensor([[0]])
result = model.generate(start, 12, generator=torch.Generator().manual_seed(1))

print("start:", start.tolist())
print("generated token IDs:", result.tolist())
print("generated length:", result.size(1))
print("Note: the untrained model has not learned meaningful transitions yet.")
