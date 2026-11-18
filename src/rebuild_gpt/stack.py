"""Run multiple causal transformer blocks in sequence."""

import torch
from torch import nn

from rebuild_gpt.block import TransformerBlock


class TransformerStack(nn.Module):
    """Apply n_layer independent pre-LayerNorm transformer blocks."""

    def __init__(self, n_embd: int, n_head: int, n_layer: int, dropout: float = 0.0,
                 attention_impl: str = "educational") -> None:
        super().__init__()
        if n_layer < 1:
            raise ValueError("n_layer must be positive")
        self.blocks = nn.ModuleList(
            [TransformerBlock(n_embd, n_head, dropout, attention_impl=attention_impl) for _ in range(n_layer)]
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            hidden = block(hidden)
        return hidden
