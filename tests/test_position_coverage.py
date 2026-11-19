"""Positional bins reconcile with the global window-token union."""

import pytest
import torch

from rebuild_gpt.eval_index_plan import EvaluationIndexPlan
from rebuild_gpt.position_coverage import analyze_position_coverage


def plan(tokens, starts, seed, block_size=3):
    return EvaluationIndexPlan(
        tokens=tokens, starts=(torch.tensor(starts, dtype=torch.long),),
        batch_size=len(starts), block_size=block_size, seed=seed,
    )


def test_uneven_spatial_coverage_and_duplicate_position_visits():
    tokens = torch.arange(20)
    report = analyze_position_coverage(
        (plan(tokens, [0, 8], 1), plan(tokens, [0, 4], 2)),
        position_bins=5,
    )
    assert [b.unique_positions for b in report.bins] == [4, 4, 4, 0, 0]
    assert [b.window_position_visits for b in report.bins] == [8, 4, 4, 0, 0]
    assert [b.repeated_position_visits for b in report.bins] == [4, 0, 0, 0, 0]
    assert report.global_unique_positions == 12
    assert report.zero_coverage_bins == 2
    assert report.global_coverage_fraction == pytest.approx(0.6)
    assert [b.coverage_fraction for b in report.bins] == [1, 1, 1, 0, 0]


def test_uneven_corpus_bins_are_never_empty():
    tokens = torch.arange(11)
    report = analyze_position_coverage(
        (plan(tokens, [0, 4], 1),), position_bins=4,
    )
    assert [(b.start, b.stop) for b in report.bins] == [
        (0, 2), (2, 5), (5, 8), (8, 11),
    ]
    assert sum(b.unique_positions for b in report.bins) == 8
    assert sum(b.window_position_visits for b in report.bins) == 8


def test_exactly_one_bin_matches_global_coverage_and_preserves_rng():
    tokens = torch.arange(40)
    before = torch.get_rng_state().clone()
    result = analyze_position_coverage(
        (plan(tokens, [1, 2, 7], 1),), position_bins=1,
    )
    assert result.bins[0].unique_positions == 9
    assert result.bins[0].window_position_visits == 12
    assert result.bins[0].repeated_position_visits == 3
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("bins", [0, -1, True, 2.2, "4", 21, 257])
def test_rejects_invalid_bin_counts(bins):
    with pytest.raises(ValueError, match="position_bins"):
        analyze_position_coverage(
            (plan(torch.arange(20), [0], 1),), position_bins=bins,
        )


def test_rejects_different_corpora_before_reading_coverage():
    with pytest.raises(ValueError, match="share one CPU corpus"):
        analyze_position_coverage(
            (plan(torch.arange(20), [0], 1),
             plan(torch.arange(20), [4], 2)),
            position_bins=4,
        )
