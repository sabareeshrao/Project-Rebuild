"""Pre-layer-normalized transformer blocks built from readable components."""

import torch
from torch import nn

from rebuild_gpt.feedforward import FeedForward
from rebuild_gpt.fused_attention import FusedCausalSelfAttention
from rebuild_gpt.multihead import MultiHeadAttention
from rebuild_gpt.normalization import FeatureLayerNorm


class TransformerBlock(nn.Module):
    """Causal multi-head attention and a per-token MLP, each with a residual."""

    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0,
                 attention_impl: str = "educational") -> None:
        super().__init__()
        self.ln1 = FeatureLayerNorm(n_embd)
        if attention_impl not in ("educational", "fused"):
            raise ValueError("attention_impl must be educational or fused")
        self.attention = (MultiHeadAttention(n_embd, n_head, dropout=dropout)
                          if attention_impl == "educational"
                          else FusedCausalSelfAttention(n_embd, n_head, dropout=dropout))
        self.ln2 = FeatureLayerNorm(n_embd)
        self.mlp = FeedForward(n_embd, dropout=dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attention(self.ln1(hidden))
        return hidden + self.mlp(self.ln2(hidden))
