"""Selective checkpointing uses predictable zero-based layer indices."""

import pytest

from rebuild_gpt.checkpoint_policy import checkpointed_indices


def test_none_selects_nothing_and_one_selects_every_layer():
    assert checkpointed_indices(4) == ()
    assert checkpointed_indices(4, 1) == (0, 1, 2, 3)


def test_every_second_layer_is_selected_in_one_based_order():
    assert checkpointed_indices(5, 2) == (1, 3)
    assert checkpointed_indices(6, 3) == (2, 5)
    assert checkpointed_indices(4, 4) == (3,)


@pytest.mark.parametrize("depth", [0, -1, True, 2.0, None])
def test_invalid_depth_is_rejected(depth):
    with pytest.raises(ValueError, match="n_layer"):
        checkpointed_indices(depth, 1)


@pytest.mark.parametrize("interval", [0, -1, 5, 1.2, True, False, "2"])
def test_invalid_intervals_are_rejected(interval):
    with pytest.raises(ValueError, match="every_n"):
        checkpointed_indices(4, interval)
