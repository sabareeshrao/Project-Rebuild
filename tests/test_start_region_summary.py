"""Pooled start counts can hide bad balance in each individual seed."""

import pytest
import torch

from rebuild_gpt.balanced_evaluation import prepare_balanced_evaluation_index_plan
from rebuild_gpt.eval_index_plan import EvaluationIndexPlan
from rebuild_gpt.start_region_summary import summarize_start_regions


def plan(tokens, starts, seed):
    return EvaluationIndexPlan(
        tokens=tokens, starts=(torch.tensor(starts, dtype=torch.long),),
        batch_size=len(starts), block_size=4, seed=seed,
    )


def test_pooled_balance_can_hide_complementary_empty_seed_regions():
    tokens = torch.arange(45)
    # Eligible starts: 0..40, four regions: [0,10), [10,20),
    # [20,30), [30,41). Each seed omits two entire regions.
    first = plan(tokens, [0, 1, 10, 11], 1)
    second = plan(tokens, [20, 21, 30, 31], 2)
    state = torch.get_rng_state().clone()
    report = summarize_start_regions((first, second), region_count=4)
    assert report.counts_by_seed == ((2, 2, 0, 0), (0, 0, 2, 2))
    assert report.aggregate_counts == (2, 2, 2, 2)
    assert report.aggregate_empty_regions == 0
    assert report.aggregate_imbalance == 0
    assert report.seeds_with_any_empty_region == 2
    assert report.empty_region_count_by_seed == (2, 2)
    assert report.imbalance_by_seed == (2, 2)
    assert report.maximum_seed_imbalance == 2
    assert report.window_count == 8
    assert torch.equal(state, torch.get_rng_state())


def test_balanced_per_seed_plans_also_balance_pooled_counts():
    tokens = torch.arange(210)
    plans = tuple(
        prepare_balanced_evaluation_index_plan(
            tokens, batch_size=3, block_size=6,
            eval_batches=4, region_count=5, seed=seed,
        )
        for seed in (1, 2, 3)
    )
    report = summarize_start_regions(plans, region_count=5)
    assert report.seed_count == 3
    assert report.aggregate_counts == (9, 9, 6, 6, 6)
    assert report.counts_by_seed == ((3, 3, 2, 2, 2),) * 3
    assert report.seeds_with_any_empty_region == 0
    assert report.maximum_seed_imbalance == 1


def test_single_seed_matches_per_seed_diagnostic():
    tokens = torch.arange(45)
    report = summarize_start_regions((plan(tokens, [0, 8, 35], 8),),
                                   region_count=4)
    assert report.aggregate_counts == report.per_seed[0].counts
    assert report.aggregate_imbalance == report.per_seed[0].imbalance


def test_duplicate_seeds_and_mismatched_corpus_rejected():
    tokens = torch.arange(45)
    with pytest.raises(ValueError, match="unique seeds"):
        summarize_start_regions((plan(tokens, [0], 2), plan(tokens, [10], 2)))
    with pytest.raises(ValueError, match="share one CPU corpus"):
        summarize_start_regions((
            plan(tokens, [0], 1), plan(tokens.clone(), [10], 2),
        ))


@pytest.mark.parametrize("count", [0, -1, True, 50])
def test_invalid_region_count_rejected(count):
    tokens = torch.arange(45)
    with pytest.raises(ValueError, match="region_count"):
        summarize_start_regions((plan(tokens, [0], 1),), region_count=count)
