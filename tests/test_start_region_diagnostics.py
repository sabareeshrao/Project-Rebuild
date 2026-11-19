"""Start balance diagnostics distinguish starts from token coverage."""

import pytest
import torch

from rebuild_gpt.balanced_evaluation import prepare_balanced_evaluation_index_plan
from rebuild_gpt.eval_index_plan import EvaluationIndexPlan
from rebuild_gpt.start_region_diagnostics import analyze_start_regions


def plan_with_starts(tokens, starts):
    return EvaluationIndexPlan(
        tokens=tokens,
        starts=(torch.tensor(starts, dtype=torch.long),),
        batch_size=len(starts), block_size=4, seed=2,
    )


def test_balanced_policy_has_no_empty_regions_and_at_most_one_count_gap():
    tokens = torch.arange(101)
    before = torch.get_rng_state().clone()
    plan = prepare_balanced_evaluation_index_plan(
        tokens, batch_size=3, block_size=4,
        eval_batches=4, region_count=5, seed=215,
    )
    report = analyze_start_regions(plan, region_count=5)
    assert report.counts == (3, 3, 2, 2, 2)
    assert report.sampled_window_count == 12
    assert report.empty_regions == 0
    assert report.imbalance == 1
    assert report.maximum_count == 3
    assert torch.equal(before, torch.get_rng_state())


def test_other_sampler_can_have_uneven_or_empty_regions():
    tokens = torch.arange(60)
    report = analyze_start_regions(
        plan_with_starts(tokens, [0, 1, 2, 10]),
        region_count=4,
    )
    assert report.counts == (4, 0, 0, 0)
    assert report.empty_regions == 3
    assert report.imbalance == 4


def test_region_count_is_not_a_token_coverage_guarantee():
    tokens = torch.arange(21)
    plan = prepare_balanced_evaluation_index_plan(
        tokens, batch_size=4, block_size=3,
        eval_batches=1, region_count=4, seed=5,
    )
    starts = analyze_start_regions(plan, region_count=4)
    assert starts.empty_regions == 0
    assert starts.counts == (1, 1, 1, 1)
    # Four starts need not cover all 21 token positions with T+1 windows.
    from rebuild_gpt.window_overlap import analyze_index_plan_overlap
    coverage = analyze_index_plan_overlap(plan)
    assert coverage.unique_corpus_token_positions <= 16 < 21


def test_out_of_range_positions_and_bad_dtypes_rejected():
    tokens = torch.arange(40)
    for values in ([36], [-1]):
        with pytest.raises(ValueError, match="out-of-range"):
            analyze_start_regions(plan_with_starts(tokens, values))
    bad = EvaluationIndexPlan(
        tokens=tokens,
        starts=(torch.tensor([0], dtype=torch.int32),),
        batch_size=1, block_size=4, seed=2,
    )
    with pytest.raises(ValueError, match="CPU int64"):
        analyze_start_regions(bad)


@pytest.mark.parametrize("region_count", [0, 100, True, 1.2])
def test_region_argument_rejected(region_count):
    with pytest.raises(ValueError, match="region_count"):
        analyze_start_regions(
            plan_with_starts(torch.arange(40), [0]),
            region_count=region_count,
        )
