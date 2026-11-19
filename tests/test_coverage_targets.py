"""Coverage targets report the first crossing, not statistical confidence."""

import pytest
import torch

from rebuild_gpt.coverage_growth import analyze_coverage_growth
from rebuild_gpt.coverage_targets import assess_coverage_target
from rebuild_gpt.eval_index_plan import EvaluationIndexPlan


def curve():
    tokens = torch.arange(20)
    plans = [
        EvaluationIndexPlan(
            tokens=tokens,
            starts=(torch.tensor(starts),),
            batch_size=len(starts), block_size=3, seed=seed,
        )
        for seed, starts in [(10, [0, 8]), (11, [0, 4]), (12, [0, 0])]
    ]
    return analyze_coverage_growth(plans)


def test_fractional_target_crossed_by_second_seed():
    report = assess_coverage_target(
        curve(), corpus_token_count=20, target_fraction=0.6,
    )
    assert report.target_token_positions == 12
    assert report.reached
    assert report.first_reaching_seed_number == 2
    assert report.first_reaching_seed == 11
    assert report.final_unique_positions == 12
    assert report.additional_positions_needed == 0
    assert report.observed_seed_count == 3


def test_small_target_crossed_by_first_seed():
    result = assess_coverage_target(
        curve(), corpus_token_count=20, target_fraction=0.4,
    )
    assert result.reached
    assert result.first_reaching_seed_number == 1


def test_unreached_target_reports_missing_positions_without_extra_sampling():
    steps = curve()
    result = assess_coverage_target(
        steps, corpus_token_count=20, target_fraction=1,
    )
    assert result.reached is False
    assert result.first_reaching_seed_number is None
    assert result.first_reaching_seed is None
    assert result.additional_positions_needed == 8
    assert len(steps) == 3


def test_noninteger_token_target_is_rounded_up():
    result = assess_coverage_target(
        curve(), corpus_token_count=20, target_fraction=0.55,
    )
    assert result.target_token_positions == 11
    assert result.first_reaching_seed_number == 2


@pytest.mark.parametrize("value", [0, -0.1, 1.01, True, "0.6",
                                  float("nan"), float("inf")])
def test_bad_targets_are_rejected(value):
    with pytest.raises(ValueError, match="target_fraction"):
        assess_coverage_target(
            curve(), corpus_token_count=20, target_fraction=value,
        )


def test_invalid_corpus_size_and_curves_are_rejected():
    with pytest.raises(ValueError, match="corpus_token_count"):
        assess_coverage_target(curve(), corpus_token_count=0, target_fraction=0.5)
    with pytest.raises(ValueError, match="steps"):
        assess_coverage_target([], corpus_token_count=20, target_fraction=0.5)
    with pytest.raises(ValueError, match="ordered"):
        assess_coverage_target(
            list(reversed(curve())), corpus_token_count=20, target_fraction=0.5,
        )
