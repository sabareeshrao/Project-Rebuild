"""Lesson 04: turn a sentence into IDs and back."""

from rebuild_gpt.tokenizer import CharTokenizer

text = "hello, GPT!"
tokenizer = CharTokenizer(text)
ids = tokenizer.encode(text)

print("text:", text)
print("IDs:", ids)
print("decoded:", tokenizer.decode(ids))
assert tokenizer.decode(ids) == text
