"""One shared linear operation to produce all attention heads' Q, K and V."""

import torch
from torch import nn


class FusedQKVProjection(nn.Module):
    """Project [B,T,C] into three [B,H,T,D] tensors, where H*D=C."""

    def __init__(self, n_embd: int, n_head: int, *, bias: bool = True) -> None:
        super().__init__()
        if n_embd < 1 or n_head < 1 or n_embd % n_head != 0:
            raise ValueError("n_embd must be positive and divisible by positive n_head")
        self.n_embd = n_embd
        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (hidden.ndim != 3 or hidden.numel() == 0
                or not hidden.is_floating_point() or hidden.size(-1) != self.n_embd):
            raise ValueError("hidden must be floating-point [B,T,n_embd]")
        batch, time, _ = hidden.shape
        packed = self.c_attn(hidden).reshape(batch, time, 3, self.n_head, self.head_size)
        query, key, value = packed.unbind(dim=2)
        return query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
