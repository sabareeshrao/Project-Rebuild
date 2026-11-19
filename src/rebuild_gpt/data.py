"""Read text and prepare contiguous token streams for GPT training."""

from pathlib import Path

import torch

from rebuild_gpt.tokenizer import CharTokenizer
from rebuild_gpt.tensors import next_token_pairs


def load_corpus(path: str | Path) -> tuple[torch.Tensor, CharTokenizer]:
    """Read a UTF-8 file and encode every character as a torch.long token."""
    text = Path(path).read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    return tokens, tokenizer


def split_train_val(
    tokens: torch.Tensor, val_fraction: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the earlier contiguous tokens for training, the rest for validation."""
    if tokens.ndim != 1:
        raise ValueError("Expected a flat [tokens] tensor")
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    split_at = int(len(tokens) * (1 - val_fraction))
    if split_at < 2 or len(tokens) - split_at < 2:
        raise ValueError("Both training and validation require at least two tokens")
    return tokens[:split_at], tokens[split_at:]


def sample_batch(
    tokens: torch.Tensor,
    *,
    batch_size: int,
    block_size: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random overlapping contexts and their next-token targets."""
    if tokens.ndim != 1 or tokens.dtype != torch.long:
        raise ValueError("Expected a flat torch.long token stream")
    if batch_size < 1 or block_size < 1:
        raise ValueError("batch_size and block_size must be positive")
    if len(tokens) <= block_size:
        raise ValueError("Need at least block_size + 1 tokens")

    starts = torch.randint(
        0, len(tokens) - block_size, (batch_size,), generator=generator, device=tokens.device
    )
    offsets = torch.arange(block_size + 1, device=tokens.device)
    windows = tokens[starts[:, None] + offsets]
    return next_token_pairs(windows)


def gather_next_token_pairs(
    tokens: torch.Tensor,
    starts: torch.Tensor,
    *,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather fixed random-start contexts, without drawing any RNG samples.

    This separates random index generation from window gathering so
    different sampling implementations can be benchmarked fairly.
    """
    if tokens.ndim != 1 or tokens.dtype != torch.long:
        raise ValueError("Expected a flat torch.long token stream")
    if (
        starts.ndim != 1 or starts.dtype != torch.long
        or starts.numel() < 1 or starts.device != tokens.device
    ):
        raise ValueError("starts must be a nonempty torch.long vector on tokens device")
    if type(block_size) is not int or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if len(tokens) <= block_size:
        raise ValueError("Need at least block_size + 1 tokens")
    if bool((starts < 0).any()) or bool((starts >= len(tokens) - block_size).any()):
        raise ValueError("starts must refer to complete next-token windows")
    offsets = torch.arange(block_size + 1, device=tokens.device)
    return next_token_pairs(tokens[starts[:, None] + offsets])


def gather_unfold_next_token_pairs(
    tokens: torch.Tensor,
    starts: torch.Tensor,
    *,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather next-token windows from an overlapping unfold *view*.

    No full corpus-sized [N,T] matrix is materialized by unfold. The
    indexed result copies only the requested batch. This is an
    alternative experiment, not a guaranteed sampling speedup.
    """
    if tokens.ndim != 1 or tokens.dtype != torch.long:
        raise ValueError("Expected a flat torch.long token stream")
    if type(block_size) is not int or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if len(tokens) <= block_size:
        raise ValueError("Need at least block_size + 1 tokens")
    if (
        starts.ndim != 1 or starts.dtype != torch.long
        or starts.numel() == 0 or starts.device != tokens.device
    ):
        raise ValueError("starts must be a nonempty torch.long vector on tokens device")
    if bool((starts < 0).any()) or bool((starts >= len(tokens) - block_size).any()):
        raise ValueError("starts must refer to complete next-token windows")
    windows_view = tokens.unfold(0, block_size + 1, 1)
    return next_token_pairs(windows_view[starts])
