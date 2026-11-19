"""Between-seed duplication is not the same as overlap within seeds."""

import pytest
import torch

from rebuild_gpt.eval_index_plan import EvaluationIndexPlan
from rebuild_gpt.cross_seed_overlap import analyze_cross_seed_overlap


def plan(tokens, starts, seed, block_size=3):
    return EvaluationIndexPlan(
        tokens=tokens,
        starts=(torch.tensor(starts, dtype=torch.long),),
        batch_size=len(starts), block_size=block_size, seed=seed,
    )


def test_disjoint_within_seeds_can_overlap_between_seeds():
    tokens = torch.arange(20)
    first = plan(tokens, [0, 8], 1)
    second = plan(tokens, [0, 4], 2)
    result = analyze_cross_seed_overlap((first, second))
    assert result.seed_count == 2
    assert result.window_count == 4
    assert result.unique_start_count == 3
    assert result.duplicate_starts_within_seeds == 0
    assert result.duplicate_starts_between_seeds == 1
    assert result.unique_corpus_token_positions == 12
    assert result.repeated_positions_within_seeds == 0
    assert result.repeated_positions_between_seeds == 4
    assert result.total_window_token_positions == 16
    assert result.total_repeated_positions == 4
    assert result.coverage_fraction == pytest.approx(0.6)


def test_overlap_within_and_between_seeds_account_for_all_repeats():
    tokens = torch.arange(20)
    first = plan(tokens, [0, 0], 1)
    second = plan(tokens, [0, 8], 2)
    result = analyze_cross_seed_overlap((first, second))
    assert result.duplicate_starts_within_seeds == 1
    assert result.duplicate_starts_between_seeds == 1
    assert result.repeated_positions_within_seeds == 4
    assert result.repeated_positions_between_seeds == 4
    assert result.unique_corpus_token_positions == 8
    assert result.total_repeated_positions == (
        result.total_window_token_positions - result.unique_corpus_token_positions
    )


def test_one_plan_has_no_between_seed_reuse_and_does_not_consume_rng():
    tokens = torch.arange(20)
    sample = plan(tokens, [1, 2], 1)
    state = torch.get_rng_state().clone()
    result = analyze_cross_seed_overlap((sample,))
    assert result.repeated_positions_between_seeds == 0
    assert result.duplicate_starts_between_seeds == 0
    assert result.repeated_positions_within_seeds == 3
    assert torch.equal(state, torch.get_rng_state())


def test_seed_identity_and_shared_corpus_are_required():
    tokens = torch.arange(20)
    original = plan(tokens, [0], 1)
    with pytest.raises(ValueError, match="unique seeds"):
        analyze_cross_seed_overlap((original, plan(tokens, [8], 1)))
    with pytest.raises(ValueError, match="share one CPU corpus"):
        analyze_cross_seed_overlap((original, plan(tokens.clone(), [8], 2)))
    with pytest.raises(ValueError, match="share one CPU corpus"):
        analyze_cross_seed_overlap((original, plan(tokens, [8], 2, block_size=4)))


@pytest.mark.parametrize("invalid", [[], None, "plans", [123]])
def test_invalid_plans_rejected(invalid):
    with pytest.raises(ValueError, match="plans"):
        analyze_cross_seed_overlap(invalid)
