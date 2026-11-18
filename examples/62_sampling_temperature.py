"""Lesson 65: temperature controls how strongly we prefer high-logit tokens."""

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.sampling import scale_logits

logits = torch.tensor([[0.0, 1.0, 3.0]])
for temperature in (0.5, 1.0, 2.0):
    probabilities = scale_logits(logits, temperature).softmax(dim=-1)
    print(f"temperature={temperature}: probabilities={probabilities[0].tolist()}")

torch.manual_seed(5)
model = GPTLanguageModel(GPTConfig(7, 6, 8, 2, 1, attention_impl="fused")).eval()
print("generated IDs:", model.generate(
    torch.tensor([[1]]), 5, temperature=0.7,
    generator=torch.Generator().manual_seed(9)
).tolist())
