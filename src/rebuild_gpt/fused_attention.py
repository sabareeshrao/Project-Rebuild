"""GPT-style causal multi-head attention with a packed QKV projection."""

import torch
from torch import nn
from torch.nn import functional as F

from rebuild_gpt.qkv import FusedQKVProjection


class FusedCausalSelfAttention(nn.Module):
    """Use [B,H,T,D] batched attention instead of independent Python heads."""

    def __init__(
        self, n_embd: int, n_head: int, dropout: float = 0.0,
        *, bias: bool = True, use_sdpa: bool = False,
    ) -> None:
        super().__init__()
        if not 0 <= dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")
        if type(use_sdpa) is not bool:
            raise ValueError("use_sdpa must be boolean")
        self.use_sdpa = use_sdpa
        self.qkv = FusedQKVProjection(n_embd, n_head, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(hidden)
        batch, heads, time, width = q.shape
        if self.use_sdpa:
            # PyTorch's SDPA chooses its backend. Dropout must be disabled
            # explicitly in eval mode; SDPA does not inspect self.training.
            mixed = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )
        else:
            # Keep the educational attention matrix visible for study.
            scores = (q @ k.transpose(-2, -1)) * (width ** -0.5)
            future = torch.ones(time, time, device=hidden.device, dtype=torch.bool).triu(1)
            weights = torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1)
            mixed = self.attn_dropout(weights) @ v
        mixed = mixed.transpose(1, 2).contiguous().view(batch, time, heads * width)
        return self.resid_dropout(self.c_proj(mixed))
