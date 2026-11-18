"""Lesson 03: build a stable mapping between characters and token IDs."""

from rebuild_gpt.tokenizer import CharTokenizer

tokenizer = CharTokenizer("cab cab!")
print("vocabulary:", tokenizer.vocab)
print("character -> ID:", tokenizer.stoi)
print("vocabulary size:", tokenizer.vocab_size)
