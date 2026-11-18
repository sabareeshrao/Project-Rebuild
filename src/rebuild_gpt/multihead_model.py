"""A small multi-head causal language model, prior to full transformer blocks."""

import torch
from torch import nn

from rebuild_gpt.embeddings import PositionEmbedding, TokenEmbedding, VocabularyProjection
from rebuild_gpt.multihead import MultiHeadAttention
from rebuild_gpt.probability import next_token_cross_entropy


class MultiHeadLanguageModel(nn.Module):
    """Add multi-head causal context to token/position vectors with a residual."""

    def __init__(
        self, vocab_size: int, block_size: int, n_embd: int,
        n_head: int, dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.tokens = TokenEmbedding(vocab_size, n_embd)
        self.positions = PositionEmbedding(block_size, n_embd)
        self.attention = MultiHeadAttention(n_embd, n_head, dropout=dropout)
        self.head = VocabularyProjection(n_embd, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.tokens(token_ids)
        hidden = hidden + self.positions(token_ids.size(1), device=token_ids.device)
        hidden = hidden + self.attention(hidden)  # residual connection
        return self.head(hidden)

    def loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return next_token_cross_entropy(self(inputs), targets)
