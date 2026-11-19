"""Regional coverage thresholds can catch blind spots in global coverage."""

import pytest
import torch

from rebuild_gpt.eval_index_plan import EvaluationIndexPlan
from rebuild_gpt.position_growth import analyze_position_growth
from rebuild_gpt.position_targets import assess_position_target


def curve(starts_by_seed):
    tokens = torch.arange(20)
    plans = tuple(
        EvaluationIndexPlan(
            tokens=tokens, starts=(torch.tensor(starts, dtype=torch.long),),
            batch_size=len(starts), block_size=3, seed=seed,
        )
        for seed, starts in enumerate(starts_by_seed, start=10)
    )
    return analyze_position_growth(plans, position_bins=5)


def test_global_majority_can_leave_two_bins_entirely_uncovered():
    result = assess_position_target(
        curve([[0, 8], [0, 4]]),
        corpus_token_count=20, minimum_bin_fraction=0.5,
    )
    assert result.reached is False
    assert result.first_reaching_seed is None
    assert result.final_positions_by_bin == (4, 4, 4, 0, 0)
    assert result.required_positions_by_bin == (2, 2, 2, 2, 2)
    assert result.missing_positions_by_bin == (0, 0, 0, 2, 2)
    assert result.bins_below_target == 2


def test_first_reaching_seed_is_detected_from_fixed_preselected_sequence():
    steps = curve([[0, 8], [4, 12, 16], [0, 8]])
    result = assess_position_target(
        steps, corpus_token_count=20, minimum_bin_fraction=1,
    )
    assert result.reached is True
    assert result.first_reaching_seed_number == 2
    assert result.first_reaching_seed == 11
    assert result.missing_positions_by_bin == (0, 0, 0, 0, 0)
    assert result.bins_below_target == 0
    assert result.observed_seed_count == 3


def test_unequal_size_bins_have_individual_integer_targets():
    tokens = torch.arange(11)
    plan = EvaluationIndexPlan(
        tokens=tokens,
        starts=(torch.tensor([0, 4]),),
        batch_size=2, block_size=3, seed=1,
    )
    report = assess_position_target(
        analyze_position_growth((plan,), position_bins=4),
        corpus_token_count=11, minimum_bin_fraction=0.5,
    )
    assert report.required_positions_by_bin == (1, 2, 2, 2)
    assert report.bins_below_target >= 1


@pytest.mark.parametrize("invalid", [
    0, -0.1, 1.1, True, "0.5", float("nan"), float("inf"),
])
def test_invalid_minimum_fraction_rejected(invalid):
    with pytest.raises(ValueError, match="minimum_bin_fraction"):
        assess_position_target(
            curve([[0]]), corpus_token_count=20,
            minimum_bin_fraction=invalid,
        )


def test_empty_or_reordered_curves_rejected():
    with pytest.raises(ValueError, match="steps"):
        assess_position_target(
            [], corpus_token_count=20, minimum_bin_fraction=0.5,
        )
    with pytest.raises(ValueError, match="ordered"):
        assess_position_target(
            list(reversed(curve([[0], [4]]))),
            corpus_token_count=20, minimum_bin_fraction=0.5,
        )
