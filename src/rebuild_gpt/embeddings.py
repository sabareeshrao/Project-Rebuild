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


class VocabularyProjection(nn.Module):
    """Project each C-dimensional vector into V next-token logits."""

    def __init__(self, n_embd: int, vocab_size: int) -> None:
        super().__init__()
        if n_embd < 1 or vocab_size < 1:
            raise ValueError("n_embd and vocab_size must be positive")
        self.linear = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        if (
            hidden.ndim != 3
            or hidden.numel() == 0
            or not hidden.is_floating_point()
            or hidden.size(-1) != self.linear.in_features
        ):
            raise ValueError("hidden must be floating-point [batch, time, n_embd]")
        return self.linear(hidden)
