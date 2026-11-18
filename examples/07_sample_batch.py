"""Lesson 07: draw random fixed-length next-token prediction windows."""

import torch

from rebuild_gpt.data import load_corpus, sample_batch, split_train_val

stream, tokenizer = load_corpus("data/tiny_story.txt")
train, _ = split_train_val(stream, val_fraction=0.2)
x, y = sample_batch(
    train, batch_size=4, block_size=8, generator=torch.Generator().manual_seed(42)
)

print("input shape:", tuple(x.shape))  # (4, 8)
print("target shape:", tuple(y.shape))  # (4, 8)
print("input example:", repr(tokenizer.decode(x[0].tolist())))
print("target example:", repr(tokenizer.decode(y[0].tolist())))
