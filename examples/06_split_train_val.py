"""Lesson 06: keep the held-out validation text separate from training."""

from rebuild_gpt.data import load_corpus, split_train_val

tokens, tokenizer = load_corpus("data/tiny_story.txt")
train, val = split_train_val(tokens, val_fraction=0.2)
print("training tokens:", len(train))
print("validation tokens:", len(val))
print("validation excerpt:", repr(tokenizer.decode(val[:40].tolist())))
