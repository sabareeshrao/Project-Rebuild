"""A grouped resume point must store optimizer moments and RNG state."""

import pytest
import torch

from rebuild_gpt.resume_inspection import inspect_resume_checkpoint
from rebuild_gpt.text_training import train_char_gpt


def _checkpoint(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abba" * 50, encoding="utf-8")
    checkpoint = tmp_path / "resume.pt"
    train_char_gpt(
        corpus, steps=2, block_size=3, n_embd=8, n_head=2, n_layer=1,
        batch_size=2, seed=7, grouped_adamw=True,
        training_state_path=checkpoint,
    )
    return checkpoint


def test_inspection_reports_training_step_and_all_moments(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    before_rng = torch.get_rng_state().clone()
    info = inspect_resume_checkpoint(checkpoint)
    assert info["step"] == 2
    assert info["optimizer_groups"] == 2
    assert info["optimizer_parameters"] == info["moment_parameters"] > 0
    assert info["batch_rng_bytes"] > 0
    assert info["cpu_rng_bytes"] > 0
    assert len(info["corpus_sha256"]) == 64
    assert torch.equal(torch.get_rng_state(), before_rng)


def test_inspection_rejects_corrupt_optimizer_state(tmp_path):
    checkpoint = _checkpoint(tmp_path)
    state = torch.load(checkpoint, weights_only=True)
    state["optimizer"]["state"].clear()
    broken = tmp_path / "broken.pt"
    torch.save(state, broken)
    with pytest.raises(ValueError, match="do not cover"):
        inspect_resume_checkpoint(broken)


def test_unsupported_format_rejected(tmp_path):
    path = tmp_path / "other.pt"
    torch.save({"format_version": -1}, path)
    with pytest.raises(ValueError, match="Unsupported"):
        inspect_resume_checkpoint(path)
