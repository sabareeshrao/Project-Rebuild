"""Find exact reasons for checkpoint drift without retraining models."""

import torch

from rebuild_gpt.resume_diagnostics import (
    diagnose_resume_difference, first_state_difference,
)


def test_nested_tensor_difference_has_named_location():
    before = {"optimizer": {"state": {0: {"exp_avg": torch.tensor([1.0, 2.0])}}}}
    after = {"optimizer": {"state": {0: {"exp_avg": torch.tensor([1.0, 3.0])}}}}
    assert first_state_difference(before, before) is None
    assert first_state_difference(before, after) == (
        "checkpoint.optimizer.state.0.exp_avg"
    )


def test_missing_key_and_changed_sequence_length_have_paths():
    assert first_state_difference({"a": 1}, {"b": 1}) == "checkpoint.keys"
    assert first_state_difference({"a": [1]}, {"a": [1, 2]}) == (
        "checkpoint.a.length"
    )


def test_diagnosis_detects_rng_change_and_preserves_original_rng(tmp_path):
    first = tmp_path / "a.pt"
    second = tmp_path / "b.pt"
    state = {
        "batch_generator_state": torch.arange(8, dtype=torch.uint8),
        "optimizer": {"state": {0: {"step": torch.tensor(3.0)}}},
    }
    torch.save(state, first)
    torch.save(state, second)
    before_rng = torch.get_rng_state().clone()
    assert diagnose_resume_difference(first, second) == {
        "identical": True, "first_difference": None,
    }

    modified = torch.load(second, weights_only=True)
    modified["batch_generator_state"][2] = 111
    torch.save(modified, second)
    report = diagnose_resume_difference(first, second)
    assert report == {
        "identical": False,
        "first_difference": "checkpoint.batch_generator_state",
    }
    assert torch.equal(before_rng, torch.get_rng_state())


def test_tied_weights_and_dtype_changes_are_not_silently_ignored():
    assert first_state_difference(
        {"model": {"weight": torch.tensor([1], dtype=torch.long)}},
        {"model": {"weight": torch.tensor([1.0])}},
    ) == "checkpoint.model.weight"
