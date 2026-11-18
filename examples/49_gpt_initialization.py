"""Lesson 49: inspect nanoGPT-style initialization and residual scaling."""

import math
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel

torch.manual_seed(5)
config = GPTConfig(vocab_size=30, block_size=8, n_embd=64, n_head=4, n_layer=4)
model = GPTLanguageModel(config)
block = model.blocks.blocks[0]

print("token embedding std:", round(model.tokens.lookup.weight.std().item(), 5))
print("attention residual projection std:", round(block.attention.projection.weight.std().item(), 5))
print("residual std target:", round(0.02 / math.sqrt(2 * config.n_layer), 5))
print("embeddings tied:", model.tokens.lookup.weight is model.head.linear.weight)
