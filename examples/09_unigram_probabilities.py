"""Lesson 09: frequencies in the training data become probabilities."""

from rebuild_gpt.baselines import unigram_probabilities
from rebuild_gpt.data import load_corpus, split_train_val

tokens, tokenizer = load_corpus("data/tiny_story.txt")
train, _ = split_train_val(tokens, val_fraction=0.2)
probs = unigram_probabilities(train, tokenizer.vocab_size)
print("probabilities sum to:", probs.sum().item())
for char, probability in zip(tokenizer.vocab, probs.tolist()):
    print(repr(char), f"{probability:.3f}")
