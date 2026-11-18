"""Safely persist the complete state of a CPU character-GPT training run.

Inference checkpoints in text_checkpoint.py are deliberately separate: they
contain neither optimizer state nor the random streams required for resuming.
"""

from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import torch

from rebuild_gpt.tokenizer import CharTokenizer


TEXT_TRAINING_STATE_VERSION = 1


def corpus_digest(path: str | Path) -> str:
    """Detect changes to the exact corpus bytes, including line endings."""
    return sha256(Path(path).read_bytes()).hexdigest()


def save_text_training_state(
    path: str | Path,
    *,
    model: torch.nn.Module,
    tokenizer: CharTokenizer,
    optimizer: torch.optim.Optimizer,
    corpus_path: str | Path,
    controls: dict,
    step: int,
    batch_generator: torch.Generator,
    history: list[dict],
    validation_before: float,
    last_loss: float,
    last_learning_rate: float,
    best_state: dict | None = None,
) -> None:
    """Save weights, AdamW internals, RNG states, and progress metadata."""
    if type(step) is not int or step < 0:
        raise ValueError("step must be a nonnegative integer")
    if model.config.vocab_size != tokenizer.vocab_size:
        raise ValueError("model vocabulary does not match tokenizer")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": TEXT_TRAINING_STATE_VERSION,
        "config": asdict(model.config),
        "vocab": list(tokenizer.vocab),
        "corpus_sha256": corpus_digest(corpus_path),
        "controls": controls,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "cpu_rng_state": torch.get_rng_state(),
        "batch_generator_state": batch_generator.get_state(),
        "history": list(history),
        "validation_before": float(validation_before),
        "last_loss": float(last_loss),
        "last_learning_rate": float(last_learning_rate),
        "best_state": best_state,
    }, output)


def load_text_training_state(
    path: str | Path,
    *,
    model: torch.nn.Module,
    tokenizer: CharTokenizer,
    corpus_path: str | Path,
    controls: dict,
) -> dict:
    """Check identity before the caller changes model or optimizer state.

    Only load trusted checkpoint files. weights_only=True restricts Python
    object deserialization, but does not make arbitrary downloads trusted.
    """
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(saved, dict) or saved.get("format_version") != TEXT_TRAINING_STATE_VERSION:
        raise ValueError("Unsupported text training state format_version")
    expected = {
        "config": asdict(model.config),
        "vocab": list(tokenizer.vocab),
        "corpus_sha256": corpus_digest(corpus_path),
        "controls": controls,
    }
    for key, value in expected.items():
        if saved.get(key) != value:
            raise ValueError(f"Checkpoint {key} does not match current training setup")
    if type(saved.get("step")) is not int or saved["step"] < 0:
        raise ValueError("Checkpoint step must be nonnegative")
    return saved
