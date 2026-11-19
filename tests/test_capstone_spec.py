"""The capstone must reject inconsistent settings before using compute."""

import pytest

from rebuild_gpt.capstone_spec import CapstoneSpec


def test_capstone_defaults_are_small_and_cpu_only():
    spec = CapstoneSpec()
    assert spec.training_kwargs() == {
        "steps": 4, "block_size": 4, "n_embd": 8, "n_head": 2,
        "n_layer": 1, "batch_size": 4, "learning_rate": 0.03,
        "seed": 7, "device": "cpu",
    }


@pytest.mark.parametrize("changes", [
    {"steps": 0}, {"n_head": 3}, {"seed": -1},
    {"prompt": ""}, {"new_tokens": -1},
    {"temperature": 0}, {"learning_rate": float("nan")}, {"top_k": 0},
])
def test_capstone_rejects_bad_settings(changes):
    with pytest.raises(ValueError):
        CapstoneSpec(**changes)
