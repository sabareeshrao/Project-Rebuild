"""A perfect start histogram does not imply every corpus-token bin is visited."""

import pytest
import torch

from rebuild_gpt.eval_index_plan import EvaluationIndexPlan
from rebuild_gpt.balanced_evaluation import prepare_balanced_evaluation_index_plan
from rebuild_gpt.sampling_policy_audit import audit_start_and_token_coverage


def plan(tokens, starts, seed):
    return EvaluationIndexPlan(
        tokens=tokens,
        starts=(torch.tensor(starts, dtype=torch.long),),
        batch_size=len(starts), block_size=3, seed=seed,
    )


def test_every_start_region_sampled_yet_many_token_bins_untouched():
    tokens = torch.arange(60)
    # Eligible starts 0..56, four start regions. Four sample starts
    # each cover only four tokens; ten positional bins partition 60.
    samples = (plan(tokens, [0, 14, 28, 42], 1),)
    rng = torch.get_rng_state().clone()
    report = audit_start_and_token_coverage(
        samples, start_regions=4, token_bins=10,
    )
    assert report.start_regions.aggregate_counts == (1, 1, 1, 1)
    assert report.start_regions.maximum_seed_imbalance == 0
    assert report.all_seeds_touch_every_start_region
    assert report.any_unsampled_token_bin
    assert report.start_balance_without_full_token_bin_coverage
    assert report.token_positions.global_unique_positions == 16
    assert report.token_positions.zero_coverage_bins >= 4
    assert report.cross_seed_overlap.unique_corpus_token_positions == 16
    assert torch.equal(rng, torch.get_rng_state())


def test_complete_token_coverage_does_not_require_multiple_start_regions():
    tokens = torch.arange(20)
    # Five adjacent disjoint windows cover the entire corpus.
    samples = (plan(tokens, [0, 4, 8, 12, 16], 1),)
    report = audit_start_and_token_coverage(
        samples, start_regions=4, token_bins=5,
    )
    assert report.token_positions.global_unique_positions == 20
    assert report.token_positions.zero_coverage_bins == 0
    assert not report.any_unsampled_token_bin
    assert not report.start_balance_without_full_token_bin_coverage


def test_balanced_sample_invariants_reconcile_with_union():
    tokens = torch.arange(150)
    plans = tuple(prepare_balanced_evaluation_index_plan(
        tokens, batch_size=3, block_size=3, eval_batches=2,
        region_count=3, seed=seed,
    ) for seed in (1, 2))
    report = audit_start_and_token_coverage(
        plans, start_regions=3, token_bins=8,
    )
    assert report.start_regions.aggregate_counts == (4, 4, 4)
    assert report.start_regions.window_count == 12
    assert report.cross_seed_overlap.window_count == 12
    assert report.token_positions.global_unique_positions == (
        report.cross_seed_overlap.unique_corpus_token_positions
    )
    assert all(x == (2, 2, 2) for x in report.start_regions.counts_by_seed)


@pytest.mark.parametrize("start_regions,token_bins", [
    (0, 4), (4, 0), (30, 4), (4, 61), (True, 4),
])
def test_invalid_partition_counts_rejected(start_regions, token_bins):
    tokens = torch.arange(60)
    with pytest.raises(ValueError, match="region_count|position_bins"):
        audit_start_and_token_coverage(
            (plan(tokens, [0], 1),),
            start_regions=start_regions, token_bins=token_bins,
        )


def test_duplicate_seed_and_mismatched_corpus_rejected():
    tokens = torch.arange(60)
    with pytest.raises(ValueError, match="unique seeds"):
        audit_start_and_token_coverage((
            plan(tokens, [0], 1), plan(tokens, [4], 1),
        ))
    with pytest.raises(ValueError, match="share one CPU corpus"):
        audit_start_and_token_coverage((
            plan(tokens, [0], 1), plan(tokens.clone(), [4], 2),
        ))
