"""Statistics are descriptive and well-defined for small samples."""

import math

import pytest

from rebuild_gpt.evaluation_stats import summarize_losses


def test_known_sample_statistics():
    stats = summarize_losses([1.0, 2.0, 3.0])
    assert stats.count == 3
    assert stats.mean == pytest.approx(2.0)
    assert stats.sample_stdev == pytest.approx(1.0)
    assert stats.standard_error == pytest.approx(1 / math.sqrt(3))
    assert (stats.minimum, stats.maximum) == (1, 3)


def test_one_batch_has_no_sample_standard_deviation():
    stats = summarize_losses([2.5])
    assert stats.mean == 2.5
    assert stats.sample_stdev is None and stats.standard_error is None
    assert stats.minimum == stats.maximum == 2.5


def test_signed_differences_are_accepted_and_input_unchanged():
    losses = [-2.0, 0.0, 2.0]
    assert summarize_losses(losses).mean == 0.0
    assert losses == [-2.0, 0.0, 2.0]


@pytest.mark.parametrize("bad", [
    [], "1,2", [], [float("nan")], [float("inf")],
    [1, True], [1, "2"], [None], [1, float("-inf")],
])
def test_rejects_invalid_data(bad):
    with pytest.raises(ValueError, match="finite"):
        summarize_losses(bad)
