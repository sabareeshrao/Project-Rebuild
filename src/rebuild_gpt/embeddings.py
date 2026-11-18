"""The vector representations used before transformer attention."""

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """Look up a C-dimensional vector for each token ID in [B, T]."""

    def __init__(self, vocab_size: int, n_embd: int) -> None:
        super().__init__()
        if vocab_size < 1 or n_embd < 1:
            raise ValueError("vocab_size and n_embd must be positive")
        self.vocab_size = vocab_size
        self.lookup = nn.Embedding(vocab_size, n_embd)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.numel() == 0 or tokens.dtype != torch.long:
            raise ValueError("tokens must be non-empty [batch, time] torch.long")
        if (tokens < 0).any() or (tokens >= self.vocab_size).any():
            raise ValueError("token ID outside vocabulary")
        return self.lookup(tokens)
