"""Overlap metrics distinguish exact duplicates from shared token positions."""

import pytest
import torch

from rebuild_gpt.eval_index_plan import prepare_evaluation_index_plan
from rebuild_gpt.window_overlap import (
    analyze_index_plan_overlap, analyze_window_starts,
)


def test_duplicate_and_partially_overlapping_windows_are_counted():
    stats = analyze_window_starts(
        [0, 0, 3, 10], block_size=3, corpus_token_count=20,
    )
    assert stats.window_count == 4
    assert stats.unique_start_count == 3
    assert stats.duplicate_start_count == 1
    assert stats.overlapping_neighbor_pairs == 2
    assert stats.total_window_token_positions == 16
    assert stats.unique_corpus_token_positions == 11
    assert stats.repeated_window_token_positions == 5
    assert stats.coverage_fraction == pytest.approx(11 / 20)


def test_disjoint_windows_have_zero_repeated_positions():
    stats = analyze_window_starts(
        torch.tensor([0, 4, 8]), block_size=3, corpus_token_count=16,
    )
    assert stats.unique_corpus_token_positions == 12
    assert stats.overlapping_neighbor_pairs == 0
    assert stats.repeated_window_token_positions == 0


def test_index_plan_analyzer_covers_all_batches_without_sampling_again():
    tokens = torch.arange(140)
    plan = prepare_evaluation_index_plan(
        tokens, batch_size=3, block_size=5, eval_batches=2, seed=188,
    )
    rng_state = torch.get_rng_state().clone()
    actual = analyze_index_plan_overlap(plan)
    expected = analyze_window_starts(
        torch.cat(plan.starts), block_size=5, corpus_token_count=140,
    )
    assert actual == expected
    assert actual.window_count == 6
    assert torch.equal(rng_state, torch.get_rng_state())


@pytest.mark.parametrize("values", [
    [], [-1], [17], [True], [1.2], "1,2",
    torch.tensor([1], dtype=torch.int32),
])
def test_bad_starts_rejected(values):
    with pytest.raises(ValueError, match="starts"):
        analyze_window_starts(
            values, block_size=3, corpus_token_count=20,
        )


def test_context_and_corpus_are_checked():
    with pytest.raises(ValueError, match="block_size"):
        analyze_window_starts([0], block_size=0, corpus_token_count=20)
    with pytest.raises(ValueError, match="corpus_token_count"):
        analyze_window_starts([0], block_size=4, corpus_token_count=4)
