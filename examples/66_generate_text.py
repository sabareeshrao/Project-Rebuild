"""Lesson 69: convert text into tokens, generate, and decode the result."""

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.text_generation import generate_text
from rebuild_gpt.tokenizer import CharTokenizer

text = "a tiny robot saw a star.\n"
tokenizer = CharTokenizer(text)
torch.manual_seed(7)
model = GPTLanguageModel(GPTConfig(
    tokenizer.vocab_size, block_size=16, n_embd=16, n_head=4,
    n_layer=1, attention_impl="fused",
))
result = generate_text(
    model, tokenizer, "a tiny", 20,
    temperature=0.8, top_k=5,
    generator=torch.Generator().manual_seed(10),
)
print(result)
print("Note: the example model has random, untrained weights.")
