"""Configurable stacked GPT model, assembled from the preceding lessons."""

import torch
from torch import nn

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.embeddings import PositionEmbedding, TokenEmbedding, VocabularyProjection
from rebuild_gpt.initialization import initialize_gpt_weights
from rebuild_gpt.normalization import FeatureLayerNorm
from rebuild_gpt.probability import next_token_cross_entropy
from rebuild_gpt.stack import TransformerStack


class GPTLanguageModel(nn.Module):
    """Causal decoder-only language model with configurable block depth."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.tokens = TokenEmbedding(config.vocab_size, config.n_embd)
        self.positions = PositionEmbedding(config.block_size, config.n_embd)
        self.input_dropout = nn.Dropout(config.dropout)
        self.blocks = TransformerStack(
            config.n_embd, config.n_head, config.n_layer, config.dropout,
            attention_impl=config.attention_impl
        )
        self.final_norm = FeatureLayerNorm(config.n_embd)
        self.head = VocabularyProjection(config.n_embd, config.vocab_size)
        if config.tie_embeddings:
            # Both paths use one Parameter; gradients and optimizer updates are shared.
            self.head.linear.weight = self.tokens.lookup.weight
        initialize_gpt_weights(self, config.n_layer)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.tokens(token_ids)
        hidden = self.input_dropout(
            hidden + self.positions(token_ids.size(1), device=token_ids.device)
        )
        return self.head(self.final_norm(self.blocks(hidden)))

    def loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return next_token_cross_entropy(self(inputs), targets)
