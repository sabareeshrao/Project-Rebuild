"""Lesson 12: compare two prediction baselines on held-out next tokens."""

import torch

from rebuild_gpt.baselines import uniform_logits, unigram_logits, unigram_probabilities
from rebuild_gpt.data import load_corpus, sample_batch, split_train_val
from rebuild_gpt.probability import next_token_cross_entropy

tokens, tokenizer = load_corpus("data/tiny_story.txt")
train, val = split_train_val(tokens, val_fraction=0.2)

# Learn token frequency statistics from train ONLY.
train_probs = unigram_probabilities(train, vocab_size=tokenizer.vocab_size)
# Use exactly the same validation examples for both baselines.
inputs, targets = sample_batch(
    val, batch_size=8, block_size=4, generator=torch.Generator().manual_seed(7)
)
uniform_loss = next_token_cross_entropy(uniform_logits(inputs, tokenizer.vocab_size), targets)
unigram_loss = next_token_cross_entropy(unigram_logits(inputs, train_probs), targets)

print("vocabulary size:", tokenizer.vocab_size)
print("uniform validation loss:", round(uniform_loss.item(), 4))
print("unigram validation loss:", round(unigram_loss.item(), 4))
print("lower loss = better next-token predictions on this batch")
