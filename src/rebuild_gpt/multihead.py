"""Readable multi-head attention components built from independent causal heads."""

import torch
from torch import nn

from rebuild_gpt.attention import CausalSelfAttentionHead


class ParallelAttentionHeads(nn.Module):
    """Concatenate H independent causal heads: [B,T,C] -> [B,T,H*D]."""

    def __init__(self, n_embd: int, n_head: int, head_size: int) -> None:
        super().__init__()
        if n_embd < 1 or n_head < 1 or head_size < 1:
            raise ValueError("n_embd, n_head and head_size must be positive")
        self.heads = nn.ModuleList(
            [CausalSelfAttentionHead(n_embd, head_size) for _ in range(n_head)]
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(hidden) for head in self.heads], dim=-1)
