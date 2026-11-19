"""Small, explicit settings for a reproducible CPU GPT capstone."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CapstoneSpec:
    """Keep training and sampling choices visible in one lesson-sized object."""

    steps: int = 4
    block_size: int = 4
    n_embd: int = 8
    n_head: int = 2
    n_layer: int = 1
    batch_size: int = 4
    learning_rate: float = 0.03
    seed: int = 7
    prompt: str = "a"
    new_tokens: int = 12
    temperature: float = 0.8
    top_k: int | None = 3

    def __post_init__(self) -> None:
        for name in ("steps", "block_size", "n_embd", "n_head", "n_layer", "batch_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(self.new_tokens) is not int or self.new_tokens < 0:
            raise ValueError("new_tokens must be a nonnegative integer")
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ValueError("prompt must be non-empty text")
        for name in ("learning_rate", "temperature"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if self.top_k is not None and (type(self.top_k) is not int or self.top_k < 1):
            raise ValueError("top_k must be a positive integer or None")

    def training_kwargs(self) -> dict:
        """Select only the arguments accepted by train_char_gpt."""
        return {
            "steps": self.steps,
            "block_size": self.block_size,
            "n_embd": self.n_embd,
            "n_head": self.n_head,
            "n_layer": self.n_layer,
            "batch_size": self.batch_size,
            "learning_rate": float(self.learning_rate),
            "seed": self.seed,
            "device": "cpu",
        }
