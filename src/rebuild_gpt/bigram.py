"""A minimal, learnable bigram language model."""

import torch
from torch import nn

from rebuild_gpt.probability import next_token_cross_entropy, stable_softmax


class BigramLanguageModel(nn.Module):
    """Predict the next-token logits using only the current token's ID."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        self.vocab_size = vocab_size
        self.token_to_logits = nn.Embedding(vocab_size, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map token IDs [B, T] to a separate score for each next token [B, T, V]."""
        if token_ids.ndim != 2 or token_ids.numel() == 0 or token_ids.dtype != torch.long:
            raise ValueError("token_ids must be a non-empty [batch, time] long tensor")
        if (token_ids < 0).any() or (token_ids >= self.vocab_size).any():
            raise ValueError("token ID outside vocabulary")
        return self.token_to_logits(token_ids)

    def loss(self, token_ids: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Differentiable next-token cross-entropy, averaged across B and T."""
        return next_token_cross_entropy(self(token_ids), targets)

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        num_new_tokens: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample new token IDs autoregressively, starting after each prompt."""
        if num_new_tokens < 0:
            raise ValueError("num_new_tokens must be nonnegative")
        self.forward(prompt)  # Validate shape, dtype, and token IDs, even for zero steps.
        result = prompt
        for _ in range(num_new_tokens):
            last_logits = self(result[:, -1:])[:, -1, :]
            probabilities = stable_softmax(last_logits)
            next_token = torch.multinomial(
                probabilities, num_samples=1, generator=generator
            )
            result = torch.cat((result, next_token), dim=1)
        return result
