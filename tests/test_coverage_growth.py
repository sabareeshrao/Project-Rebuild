"""Marginal gains must sum to the exact global corpus union."""

import pytest
import torch

from rebuild_gpt.coverage_growth import analyze_coverage_growth
from rebuild_gpt.cross_seed_overlap import analyze_cross_seed_overlap
from rebuild_gpt.eval_index_plan import EvaluationIndexPlan


def plan(tokens, starts, seed):
    return EvaluationIndexPlan(
        tokens=tokens,
        starts=(torch.tensor(starts, dtype=torch.long),),
        batch_size=len(starts), block_size=3, seed=seed,
    )


def test_marginal_new_coverage_accounts_for_within_and_between_seed_reuse():
    corpus = torch.arange(20)
    plans = (
        plan(corpus, [0, 8], 10),
        plan(corpus, [0, 4], 11),
        plan(corpus, [0, 0], 12),
    )
    before = torch.get_rng_state().clone()
    steps = analyze_coverage_growth(plans)
    assert [x.seed for x in steps] == [10, 11, 12]
    assert [x.new_corpus_token_positions for x in steps] == [8, 4, 0]
    assert [x.already_covered_positions_this_seed for x in steps] == [0, 4, 4]
    assert [x.repeated_positions_within_seed for x in steps] == [0, 0, 4]
    assert [x.cumulative_unique_positions for x in steps] == [8, 12, 12]
    assert [x.cumulative_coverage_fraction for x in steps] == pytest.approx(
        [0.4, 0.6, 0.6]
    )
    assert steps[-1].cumulative_unique_positions == (
        analyze_cross_seed_overlap(plans).unique_corpus_token_positions
    )
    assert sum(x.new_corpus_token_positions for x in steps) == 12
    assert torch.equal(before, torch.get_rng_state())


def test_each_seed_decomposes_its_total_window_positions():
    tokens = torch.arange(100)
    plans = (
        plan(tokens, [0, 0, 5], 1),
        plan(tokens, [0, 6, 10], 2),
        plan(tokens, [5, 20, 30], 3),
    )
    for step in analyze_coverage_growth(plans):
        assert (
            step.new_corpus_token_positions
            + step.already_covered_positions_this_seed
            + step.repeated_positions_within_seed
            == step.window_count * 4
        )
    assert analyze_coverage_growth(plans) == analyze_coverage_growth(plans)


def test_invalid_plans_rejected_before_any_partial_curve():
    tokens = torch.arange(50)
    with pytest.raises(ValueError, match="nonempty"):
        analyze_coverage_growth([])
    with pytest.raises(ValueError, match="unique seeds"):
        analyze_coverage_growth([plan(tokens, [0], 1), plan(tokens, [4], 1)])
    with pytest.raises(ValueError, match="share one CPU corpus"):
        analyze_coverage_growth(
            [plan(tokens, [0], 1), plan(tokens.clone(), [4], 2)]
        )
