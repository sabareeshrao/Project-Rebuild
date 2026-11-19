"""Reuse a zero-copy sliding-window view for random next-token sampling."""

import torch

from rebuild_gpt.tensors import next_token_pairs


class CachedWindowSampler:
    """Cache only tensor metadata, never a full copy of all text windows.

    The underlying unfolded table overlaps the original 1-D token
    storage. Each call gathers just B selected windows. Random starts
    and generator consumption match the existing sample_batch API.
    """

    def __init__(self, tokens: torch.Tensor, *, block_size: int) -> None:
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 1 or tokens.dtype != torch.long:
            raise ValueError("Expected a flat torch.long token stream")
        if type(block_size) is not int or block_size < 1:
            raise ValueError("block_size must be a positive integer")
        if len(tokens) <= block_size:
            raise ValueError("Need at least block_size + 1 tokens")
        self.tokens = tokens
        self.block_size = block_size
        self.windows = tokens.unfold(0, block_size + 1, 1)

    def sample(
        self, *, batch_size: int, generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw exactly one vector of random starts, like sample_batch."""
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        starts = torch.randint(
            0, len(self.tokens) - self.block_size,
            (batch_size,), generator=generator, device=self.tokens.device,
        )
        return next_token_pairs(self.windows[starts])
