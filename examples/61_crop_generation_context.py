"""Lesson 64: the output keeps growing, but GPT only sees its latest context."""

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel

torch.manual_seed(5)
config = GPTConfig(9, block_size=4, n_embd=8, n_head=2, n_layer=1, attention_impl="fused")
model = GPTLanguageModel(config).eval()
long_prompt = torch.tensor([[0, 1, 2, 3, 4, 5, 6]])

out = model.generate(long_prompt, 6, generator=torch.Generator().manual_seed(1))
print("configured block_size:", config.block_size)
print("prompt length:", long_prompt.size(1))
print("total result length:", out.size(1))
print("original prompt preserved:", torch.equal(out[:, :7], long_prompt))
