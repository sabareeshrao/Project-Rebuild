"""Save and restore our own educational GPT model checkpoints safely."""

from dataclasses import asdict
from pathlib import Path

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel


MODEL_CHECKPOINT_VERSION = 1


def save_model_checkpoint(model: GPTLanguageModel, path: str | Path) -> None:
    """Write model weights and architecture settings, not arbitrary Python objects."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": MODEL_CHECKPOINT_VERSION,
        "config": asdict(model.config),
        "model": model.state_dict(),
    }, output)


def load_model_checkpoint(path: str | Path) -> GPTLanguageModel:
    """Load our format on CPU; strictly check all parameter keys and shapes."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != MODEL_CHECKPOINT_VERSION:
        raise ValueError("Unsupported model checkpoint format_version")
    config = GPTConfig(**payload["config"])
    model = GPTLanguageModel(config)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model
