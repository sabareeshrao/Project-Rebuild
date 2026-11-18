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
