"""A position-aware but still context-free next-token model."""

import torch
from torch import nn

from rebuild_gpt.embeddings import PositionEmbedding, TokenEmbedding, VocabularyProjection
from rebuild_gpt.probability import next_token_cross_entropy


class EmbeddingLanguageModel(nn.Module):
    """Predict from each token and its absolute position, not preceding tokens."""

    def __init__(self, vocab_size: int, block_size: int, n_embd: int) -> None:
        super().__init__()
        self.tokens = TokenEmbedding(vocab_size, n_embd)
        self.positions = PositionEmbedding(block_size, n_embd)
        self.head = VocabularyProjection(n_embd, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_vectors = self.tokens(token_ids)          # [B, T, C]
        position_vectors = self.positions(               # [T, C]
            token_ids.size(1), device=token_ids.device
        )
        hidden = token_vectors + position_vectors        # broadcast over B
        return self.head(hidden)                          # [B, T, V]

    def loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return next_token_cross_entropy(self(inputs), targets)
