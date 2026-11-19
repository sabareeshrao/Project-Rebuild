"""Disjoint windows must be reproducible and share no token positions."""

import pytest
import torch

from rebuild_gpt.disjoint_evaluation import prepare_disjoint_evaluation_index_plan
from rebuild_gpt.window_overlap import analyze_index_plan_overlap


def test_multiple_batches_are_disjoint_and_repeatable_without_global_rng():
    tokens = torch.arange(120, dtype=torch.long) % 11
    rng = torch.get_rng_state().clone()
    args = dict(batch_size=3, block_size=5, eval_batches=3, seed=189)
    a = prepare_disjoint_evaluation_index_plan(tokens, **args)
    b = prepare_disjoint_evaluation_index_plan(tokens, **args)
    assert a.tokens is tokens
    assert a.seed == 189
    assert all(torch.equal(x, y) for x, y in zip(a.starts, b.starts))
    assert len(list(a.batches())) == 3
    stats = analyze_index_plan_overlap(a)
    assert stats.window_count == 9
    assert stats.duplicate_start_count == 0
    assert stats.overlapping_neighbor_pairs == 0
    assert stats.repeated_window_token_positions == 0
    assert stats.unique_corpus_token_positions == 9 * 6
    assert torch.equal(rng, torch.get_rng_state())


def test_exact_tile_capacity_uses_each_tile_once():
    tokens = torch.arange(20, dtype=torch.long)
    plan = prepare_disjoint_evaluation_index_plan(
        tokens, batch_size=2, block_size=4, eval_batches=2, seed=12,
    )
    assert sorted(torch.cat(plan.starts).tolist()) == [0, 5, 10, 15]
    assert analyze_index_plan_overlap(plan).coverage_fraction == 1.0


def test_not_enough_tiles_rejected_before_random_permutation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Should fail before choosing random tiles")
    monkeypatch.setattr(torch, "randperm", forbidden)
    with pytest.raises(ValueError, match="only 3"):
        prepare_disjoint_evaluation_index_plan(
            torch.arange(19), batch_size=2, block_size=4,
            eval_batches=2, seed=189,
        )


@pytest.mark.parametrize("field,bad", [
    ("batch_size", 0), ("block_size", True), ("eval_batches", -1),
    ("eval_batches", 1.5), ("seed", -1), ("seed", False),
])
def test_invalid_arguments_are_rejected(field, bad):
    options = dict(batch_size=1, block_size=3, eval_batches=2, seed=9)
    options[field] = bad
    with pytest.raises(ValueError, match=field):
        prepare_disjoint_evaluation_index_plan(torch.arange(40), **options)


def test_wrong_token_dtype_is_rejected():
    with pytest.raises(ValueError, match="torch.long"):
        prepare_disjoint_evaluation_index_plan(
            torch.arange(40, dtype=torch.int32),
            batch_size=2, block_size=3,
        )
