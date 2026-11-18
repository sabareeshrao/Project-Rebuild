"""Restart educational AdamW training with model, optimizer, and RNG state."""

from dataclasses import asdict
from pathlib import Path

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel


TRAINING_CHECKPOINT_VERSION = 1


def save_training_checkpoint(
    model: GPTLanguageModel,
    optimizer: torch.optim.AdamW,
    step: int,
    path: str | Path,
    *,
    batch_generator: torch.Generator | None = None,
) -> None:
    """Save a CPU RNG snapshot and one-generator sampling state with AdamW."""
    if type(step) is not int or step < 0:
        raise ValueError("step must be a nonnegative integer")
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("Only AdamW training checkpoints are supported")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": TRAINING_CHECKPOINT_VERSION,
        "config": asdict(model.config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "cpu_rng_state": torch.get_rng_state(),
        "batch_generator_state": (
            batch_generator.get_state() if batch_generator is not None else None
        ),
    }, output)


def load_training_checkpoint(
    path: str | Path, *, restore_rng: bool = False
) -> tuple[GPTLanguageModel, torch.optim.AdamW, int, torch.Generator | None]:
    """Recreate model and AdamW; optionally restore the process-wide CPU RNG."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or saved.get("format_version") != TRAINING_CHECKPOINT_VERSION:
        raise ValueError("Unsupported training checkpoint format_version")
    config = GPTConfig(**saved["config"])
    model = GPTLanguageModel(config)
    model.load_state_dict(saved["model"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.load_state_dict(saved["optimizer"])
    generator = None
    if saved["batch_generator_state"] is not None:
        generator = torch.Generator(device="cpu")
        generator.set_state(saved["batch_generator_state"])
    if restore_rng:
        torch.set_rng_state(saved["cpu_rng_state"])
    model.train()
    return model, optimizer, saved["step"], generator
