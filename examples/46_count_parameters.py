"""Lesson 46: inspect how model depth changes parameter counts."""

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.parameters import count_parameters

for depth in (1, 2, 4):
    config = GPTConfig(vocab_size=30, block_size=16, n_embd=12, n_head=3, n_layer=depth)
    model = GPTLanguageModel(config)
    print(f"layers={depth}: trainable parameters={count_parameters(model):,}")
