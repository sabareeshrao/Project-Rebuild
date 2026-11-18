"""Minimal language model with a single complete causal attention head."""

import torch
from torch import nn

from rebuild_gpt.attention import CausalSelfAttentionHead
from rebuild_gpt.embeddings import PositionEmbedding, TokenEmbedding, VocabularyProjection
from rebuild_gpt.probability import next_token_cross_entropy


class SingleHeadLanguageModel(nn.Module):
    """Mix earlier-token information, then predict the next token."""

    def __init__(self, vocab_size: int, block_size: int, n_embd: int, head_size: int) -> None:
        super().__init__()
        self.tokens = TokenEmbedding(vocab_size, n_embd)
        self.positions = PositionEmbedding(block_size, n_embd)
        self.attention = CausalSelfAttentionHead(n_embd, head_size)
        self.output = nn.Linear(head_size, n_embd, bias=False)
        self.head = VocabularyProjection(n_embd, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.tokens(token_ids)
        hidden = hidden + self.positions(token_ids.size(1), device=token_ids.device)
        hidden = hidden + self.output(self.attention(hidden))  # residual skip
        return self.head(hidden)

    def loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return next_token_cross_entropy(self(inputs), targets)
