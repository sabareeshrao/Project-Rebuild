"""Lesson 66: only the top-k next-token logits remain eligible."""

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.sampling import filter_top_k

logits = torch.tensor([[1.0, 5.0, 0.0, 3.0]])
print("all candidate scores:", logits.tolist())
print("top-2 candidate scores:", filter_top_k(logits, 2).tolist())

torch.manual_seed(5)
model = GPTLanguageModel(GPTConfig(7, 5, 8, 2, 1, attention_impl="fused")).eval()
prompt = torch.tensor([[2, 1]])
print("top-1 sample:", model.generate(prompt, 8, top_k=1).tolist())
print("top-3 sample:", model.generate(
    prompt, 8, top_k=3, generator=torch.Generator().manual_seed(7)
).tolist())
