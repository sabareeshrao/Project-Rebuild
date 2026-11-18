"""A self-contained checkpoint with GPT tensors and a character vocabulary."""

from dataclasses import asdict
from pathlib import Path

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.tokenizer import CharTokenizer


TEXT_CHECKPOINT_VERSION = 1


def save_text_checkpoint(
    model: GPTLanguageModel, tokenizer: CharTokenizer, path: str | Path
) -> None:
    """Bundle tokenizer ID order, model configuration and learned weights."""
    if model.config.vocab_size != tokenizer.vocab_size:
        raise ValueError("model vocab_size must match tokenizer vocab_size")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": TEXT_CHECKPOINT_VERSION,
        "config": asdict(model.config),
        "vocab": list(tokenizer.vocab),
        "model": model.state_dict(),
    }, output)


def load_text_checkpoint(path: str | Path) -> tuple[GPTLanguageModel, CharTokenizer]:
    """Restore a self-contained inference bundle on CPU, using strict keys."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != TEXT_CHECKPOINT_VERSION:
        raise ValueError("Unsupported text checkpoint format_version")
    tokenizer = CharTokenizer.from_vocab(payload["vocab"])
    config = GPTConfig(**payload["config"])
    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError("Checkpoint vocab_size does not match saved tokenizer")
    model = GPTLanguageModel(config)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, tokenizer
