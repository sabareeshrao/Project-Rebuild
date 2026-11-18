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
            attention_impl=config.attention_impl, bias=config.bias
        )
        self.final_norm = FeatureLayerNorm(config.n_embd, bias=config.bias)
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

    def forward_nanogpt(
        self, token_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Mirror upstream GPT.forward's output contract without changing forward().

        Inference returns only the final time position. Training with targets
        returns full logits and cross-entropy ignoring target ID -1.
        """
        from torch.nn import functional as F

        logits = self(token_ids)
        if targets is None:
            return logits[:, -1:, :], None
        if targets.shape != token_ids.shape or targets.dtype != torch.long:
            raise ValueError("targets must be torch.long with shape [batch, time]")
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-1,
        )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample next tokens while keeping only the most recent context window."""
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        if (
            token_ids.ndim != 2 or token_ids.numel() == 0
            or token_ids.dtype != torch.long
        ):
            raise ValueError("token_ids must be a non-empty [batch, time] long tensor")
        if (token_ids < 0).any() or (token_ids >= self.config.vocab_size).any():
            raise ValueError("token ID outside vocabulary")

        generated = token_ids
        for _ in range(max_new_tokens):
            # The output grows without bound; the model only sees its last block_size IDs.
            context = generated[:, -self.config.block_size:]
            last_logits, _ = self.forward_nanogpt(context)
            probabilities = torch.softmax(last_logits[:, -1, :], dim=-1)
            next_id = torch.multinomial(probabilities, num_samples=1, generator=generator)
            generated = torch.cat((generated, next_id), dim=1)
        return generated

    def loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return next_token_cross_entropy(self(inputs), targets)
