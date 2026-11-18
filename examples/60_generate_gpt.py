"""Lesson 63: generate autoregressive token IDs with the stacked GPT."""

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel

torch.manual_seed(5)
model = GPTLanguageModel(GPTConfig(9, 8, 8, 2, 1, attention_impl="fused")).eval()
prompt = torch.tensor([[1, 2]])
generated = model.generate(prompt, 4, generator=torch.Generator().manual_seed(8))

print("prompt IDs:", prompt.tolist())
print("generated IDs:", generated.tolist())
print("shape:", tuple(generated.shape))
print("The model is untrained; the sample is a tensor-shape exercise.")
