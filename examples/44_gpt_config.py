"""Lesson 44: configure GPT depth, heads and feature dimensions."""

from rebuild_gpt.config import GPTConfig

config = GPTConfig(vocab_size=30, block_size=16, n_embd=12, n_head=3, n_layer=2)
print("config:", config)
print("head width (n_embd / n_head):", config.head_size)
print("maximum input sequence length:", config.block_size)
