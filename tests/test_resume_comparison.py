"""A changed AdamW moment or random stream must break resume equivalence."""

import pytest
import torch

from rebuild_gpt.resume_comparison import compare_resume_checkpoints
from rebuild_gpt.text_training import train_char_gpt


def _state(tmp_path):
    text = tmp_path / "corpus.txt"
    text.write_text("abcabc" * 30, encoding="utf-8")
    output = tmp_path / "state.pt"
    train_char_gpt(
        text, steps=2, block_size=3, n_embd=8, n_head=2, n_layer=1,
        batch_size=2, grouped_adamw=True, training_state_path=output,
    )
    return output


def test_identical_state_matches_every_tensor(tmp_path):
    file = _state(tmp_path)
    rng_before = torch.get_rng_state().clone()
    report = compare_resume_checkpoints(file, file)
    assert report["identical"]
    assert report["step"] == 2
    assert report["moment_parameters"] > 0
    assert report["matched_tensors"] > 10
    assert torch.equal(torch.get_rng_state(), rng_before)


def test_mutated_moment_is_detected_with_field_name(tmp_path):
    original = _state(tmp_path)
    data = torch.load(original, map_location="cpu", weights_only=True)
    first = next(iter(data["optimizer"]["state"]))
    data["optimizer"]["state"][first]["exp_avg"].add_(0.01)
    changed = tmp_path / "changed.pt"
    torch.save(data, changed)
    with pytest.raises(AssertionError, match="optimizer"):
        compare_resume_checkpoints(original, changed)


def test_mutated_batch_rng_is_detected(tmp_path):
    original = _state(tmp_path)
    data = torch.load(original, map_location="cpu", weights_only=True)
    data["batch_generator_state"] = torch.zeros_like(data["batch_generator_state"])
    changed = tmp_path / "rng.pt"
    torch.save(data, changed)
    with pytest.raises(AssertionError, match="batch_generator_state"):
        compare_resume_checkpoints(original, changed)
