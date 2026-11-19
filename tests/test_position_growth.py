"""Seed-level positional marginal gains reconcile with final bin coverage."""

import pytest
import torch

from rebuild_gpt.eval_index_plan import EvaluationIndexPlan
from rebuild_gpt.position_growth import analyze_position_growth
from rebuild_gpt.position_coverage import analyze_position_coverage
from rebuild_gpt.coverage_growth import analyze_coverage_growth


def plan(tokens, starts, seed):
    return EvaluationIndexPlan(
        tokens=tokens, starts=(torch.tensor(starts, dtype=torch.long),),
        batch_size=len(starts), block_size=3, seed=seed,
    )


def test_marginal_gains_show_which_positional_bins_received_new_coverage():
    tokens = torch.arange(20)
    plans = (
        plan(tokens, [0, 8], 1),
        plan(tokens, [0, 4], 2),
        plan(tokens, [0, 0], 3),
    )
    before = torch.get_rng_state().clone()
    steps = analyze_position_growth(plans, position_bins=5)
    assert [x.new_positions_by_bin for x in steps] == [
        (4, 0, 4, 0, 0),
        (0, 4, 0, 0, 0),
        (0, 0, 0, 0, 0),
    ]
    assert [x.newly_touched_bins for x in steps] == [2, 1, 0]
    assert [x.new_total_positions for x in steps] == [8, 4, 0]
    assert [x.cumulative_total_positions for x in steps] == [8, 12, 12]
    assert steps[-1].cumulative_positions_by_bin == (4, 4, 4, 0, 0)
    assert [x.new_total_positions for x in steps] == [
        x.new_corpus_token_positions for x in analyze_coverage_growth(plans)
    ]
    assert torch.equal(before, torch.get_rng_state())


def test_final_position_growth_equals_direct_profile_across_unequal_bins():
    tokens = torch.arange(23)
    plans = (
        plan(tokens, [0, 10], 10),
        plan(tokens, [3, 15], 11),
    )
    steps = analyze_position_growth(plans, position_bins=6)
    profile = analyze_position_coverage(plans, position_bins=6)
    assert steps[-1].cumulative_positions_by_bin == tuple(
        b.unique_positions for b in profile.bins
    )
    assert steps[-1].cumulative_total_positions == profile.global_unique_positions
    assert all(x >= 0 for step in steps for x in step.new_positions_by_bin)


def test_seed_order_changes_marginal_attribution_not_final_union():
    tokens = torch.arange(20)
    p, q = plan(tokens, [0, 8], 1), plan(tokens, [0, 4], 2)
    first = analyze_position_growth((p, q), position_bins=5)
    swapped = analyze_position_growth((q, p), position_bins=5)
    assert first[0].new_positions_by_bin != swapped[0].new_positions_by_bin
    assert first[-1].cumulative_positions_by_bin == (
        swapped[-1].cumulative_positions_by_bin
    )


@pytest.mark.parametrize("bad_bins", [0, 21, True, 1.2])
def test_bad_bin_count_rejected(bad_bins):
    with pytest.raises(ValueError, match="position_bins"):
        analyze_position_growth(
            (plan(torch.arange(20), [0], 1),), position_bins=bad_bins,
        )


def test_repeated_seed_rejected_before_result():
    tokens = torch.arange(20)
    with pytest.raises(ValueError, match="unique seeds"):
        analyze_position_growth(
            (plan(tokens, [0], 1), plan(tokens, [4], 1)), position_bins=5,
        )
