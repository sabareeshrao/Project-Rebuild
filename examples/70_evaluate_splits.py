"""Lesson 73: measure repeatable train and validation loss without training."""

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.data import load_corpus, split_train_val
from rebuild_gpt.evaluation import estimate_train_val_loss
from rebuild_gpt.gpt import GPTLanguageModel

tokens, tokenizer = load_corpus("data/tiny_story.txt")
train, val = split_train_val(tokens, val_fraction=0.2)
torch.manual_seed(7)
model = GPTLanguageModel(GPTConfig(tokenizer.vocab_size, 8, 16, 4, 1, attention_impl="fused"))
result = estimate_train_val_loss(model, train, val, batch_size=4, block_size=8, eval_batches=4, seed=17)
print("training estimate:", round(result["train_loss"], 4))
print("validation estimate:", round(result["validation_loss"], 4))
print("model remains in training mode:", model.training)
