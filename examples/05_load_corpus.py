"""Lesson 05: read real text and turn it into one flat sequence of IDs."""

from rebuild_gpt.data import load_corpus

tokens, tokenizer = load_corpus("data/tiny_story.txt")
print("number of tokens:", len(tokens))
print("vocabulary size:", tokenizer.vocab_size)
print("first 20 IDs:", tokens[:20].tolist())
print("first 20 characters:", tokenizer.decode(tokens[:20].tolist()))
