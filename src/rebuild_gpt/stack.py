"""Run multiple causal transformer blocks in sequence."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from rebuild_gpt.block import TransformerBlock


class TransformerStack(nn.Module):
    """Apply n_layer independent pre-LayerNorm transformer blocks."""

    def __init__(self, n_embd: int, n_head: int, n_layer: int, dropout: float = 0.0,
                 attention_impl: str = "educational", bias: bool = True,
                 activation_checkpointing: bool = False) -> None:
        super().__init__()
        if n_layer < 1:
            raise ValueError("n_layer must be positive")
        if type(activation_checkpointing) is not bool:
            raise ValueError("activation_checkpointing must be boolean")
        self.activation_checkpointing = activation_checkpointing
        self.blocks = nn.ModuleList(
            [TransformerBlock(n_embd, n_head, dropout, attention_impl=attention_impl, bias=bias) for _ in range(n_layer)]
        )

    def set_activation_checkpointing(self, enabled: bool) -> None:
        """Trade backward recomputation for fewer saved intermediate tensors."""
        if type(enabled) is not bool:
            raise ValueError("activation_checkpointing must be boolean")
        self.activation_checkpointing = enabled

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                hidden = checkpoint(
                    block, hidden, use_reentrant=False, preserve_rng_state=True
                )
            else:
                hidden = block(hidden)
        return hidden
