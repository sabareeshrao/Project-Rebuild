"""Direction summaries describe signs; they make no significance claim."""

import pytest

from rebuild_gpt.paired_diagnostics import summarize_paired_deltas


def test_known_directions_distinguish_batch_and_seed_mean_outcomes():
    report = summarize_paired_deltas([
        [-1.0, 0.0, 0.5],
        [2.0, -0.5, -0.5],
    ])
    assert report.batch_directions.improved == 3
    assert report.batch_directions.worsened == 2
    assert report.batch_directions.tied == 1
    assert report.batch_directions.total == 6
    assert report.batch_directions.improved_fraction == pytest.approx(0.5)
    assert report.seed_mean_directions.improved == 1
    assert report.seed_mean_directions.worsened == 1
    assert report.seed_mean_directions.tied == 0
    assert report.per_seed_means == pytest.approx((-1 / 6, 1 / 3))
    assert report.mean_paired_delta == pytest.approx(1 / 12)


def test_tolerance_labels_near_zero_deltas_as_ties():
    report = summarize_paired_deltas(
        [[-0.001, 0.001, 0.3]], tie_tolerance=0.01,
    )
    assert report.batch_directions.tied == 2
    assert report.batch_directions.worsened == 1
    assert report.seed_mean_directions.worsened == 1


def test_per_seed_counts_are_not_pooling_all_batches_into_independent_trials():
    report = summarize_paired_deltas([[1, -1], [1, -1], [1, -1]])
    assert report.batch_directions.tied == 0
    assert report.seed_mean_directions.tied == 3
    assert report.mean_paired_delta == 0


@pytest.mark.parametrize("invalid", [
    [], [()], [[1], []], [[1], [1, 2]],
    [[1, float("nan")]], [[1, float("inf")]], [[True]], "1",
])
def test_invalid_input_deltas_are_rejected(invalid):
    with pytest.raises(ValueError, match="per_seed_deltas"):
        summarize_paired_deltas(invalid)


@pytest.mark.parametrize("value", [-1, float("inf"), float("nan"), True, "0.1"])
def test_tie_tolerance_requires_nonnegative_finite_number(value):
    with pytest.raises(ValueError, match="tie_tolerance"):
        summarize_paired_deltas([[-1, 0, 1]], tie_tolerance=value)
