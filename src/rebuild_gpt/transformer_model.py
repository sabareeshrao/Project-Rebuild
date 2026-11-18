"""One-block GPT-style language model (before stacking many blocks)."""

import torch
from torch import nn

from rebuild_gpt.block import TransformerBlock
from rebuild_gpt.embeddings import PositionEmbedding, TokenEmbedding, VocabularyProjection
from rebuild_gpt.normalization import FeatureLayerNorm
from rebuild_gpt.probability import next_token_cross_entropy


class TransformerLanguageModel(nn.Module):
    """Token + position embeddings, one pre-norm block, final norm, logits."""

    def __init__(
        self, vocab_size: int, block_size: int, n_embd: int,
        n_head: int, dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1")
        self.tokens = TokenEmbedding(vocab_size, n_embd)
        self.positions = PositionEmbedding(block_size, n_embd)
        self.input_dropout = nn.Dropout(dropout)
        self.block = TransformerBlock(n_embd, n_head, dropout=dropout)
        self.final_norm = FeatureLayerNorm(n_embd)
        self.head = VocabularyProjection(n_embd, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.tokens(token_ids)
        hidden = self.input_dropout(
            hidden + self.positions(token_ids.size(1), device=token_ids.device)
        )
        hidden = self.block(hidden)
        return self.head(self.final_norm(hidden))

    def loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return next_token_cross_entropy(self(inputs), targets)
