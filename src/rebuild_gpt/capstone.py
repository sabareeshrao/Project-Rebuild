"""A small learning-first training entrypoint built on the existing trainer."""

from pathlib import Path

from rebuild_gpt.capstone_spec import CapstoneSpec
from rebuild_gpt.text_training import train_char_gpt


def train_capstone(corpus_path: str | Path, spec: CapstoneSpec):
    """Train a tiny real GPT; keep the existing training path authoritative."""
    return train_char_gpt(corpus_path, **spec.training_kwargs())
