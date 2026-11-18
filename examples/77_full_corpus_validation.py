"""Lesson 80: evaluate every next-token target without random sampling."""

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.corpus_evaluation import full_train_val_loss
from rebuild_gpt.data import load_corpus, split_train_val
from rebuild_gpt.gpt import GPTLanguageModel

tokens, tokenizer = load_corpus("data/tiny_story.txt")
train, validation = split_train_val(tokens, val_fraction=0.2)
torch.manual_seed(8)
model = GPTLanguageModel(GPTConfig(
    tokenizer.vocab_size, block_size=8, n_embd=16, n_head=4,
    n_layer=1, attention_impl="fused",
))
metrics = full_train_val_loss(model, train, validation, block_size=8)
print("all train transitions:", len(train) - 1, metrics["train_loss"])
print("all held-out transitions:", len(validation) - 1, metrics["validation_loss"])
print("Contexts restart at chunk boundaries; this is not sliding-window loss.")
