"""Validated hyperparameters for a small educational decoder-only GPT."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    block_size: int
    n_embd: int
    n_head: int
    n_layer: int
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in ("vocab_size", "block_size", "n_embd", "n_head", "n_layer"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if not isinstance(self.dropout, (int, float)) or not math.isfinite(self.dropout) or not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout must be finite and between 0 and 1")

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head
