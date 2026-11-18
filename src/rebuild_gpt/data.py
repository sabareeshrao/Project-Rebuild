"""Read text and prepare contiguous token streams for GPT training."""

from pathlib import Path

import torch

from rebuild_gpt.tokenizer import CharTokenizer


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
