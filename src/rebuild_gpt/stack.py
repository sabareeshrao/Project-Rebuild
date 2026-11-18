"""Run multiple causal transformer blocks in sequence."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from rebuild_gpt.block import TransformerBlock
from rebuild_gpt.checkpoint_policy import checkpointed_indices


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
        self.checkpoint_interval: int | None = None
        self.blocks = nn.ModuleList(
            [TransformerBlock(n_embd, n_head, dropout, attention_impl=attention_impl, bias=bias) for _ in range(n_layer)]
        )

    def set_activation_checkpointing(self, enabled: bool) -> None:
        """Trade backward recomputation for fewer saved intermediate tensors."""
        if type(enabled) is not bool:
            raise ValueError("activation_checkpointing must be boolean")
        self.activation_checkpointing = enabled
        self.checkpoint_interval = None

    def set_checkpoint_interval(self, every_n: int | None) -> None:
        """Choose only every Nth block; None disables recomputation."""
        checkpointed_indices(len(self.blocks), every_n)  # validate up front
        self.activation_checkpointing = False
        self.checkpoint_interval = every_n

    @property
    def checkpointed_blocks(self) -> tuple[int, ...]:
        """Zero-based block indices selected for backward recomputation."""
        if self.activation_checkpointing:
            return tuple(range(len(self.blocks)))
        return checkpointed_indices(len(self.blocks), self.checkpoint_interval)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        selected = (
            set(self.checkpointed_blocks)
            if self.training and torch.is_grad_enabled() else set()
        )
        for index, block in enumerate(self.blocks):
            if index in selected:
                hidden = checkpoint(
                    block, hidden, use_reentrant=False, preserve_rng_state=True
                )
            else:
                hidden = block(hidden)
        return hidden
