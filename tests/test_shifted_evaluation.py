"""Phase shifts change eligible starts without permitting overlap."""

import pytest
import torch

from rebuild_gpt.shifted_evaluation import (
    feasible_lattice_phases, prepare_shifted_evaluation_index_plan,
)
from rebuild_gpt.window_overlap import analyze_index_plan_overlap


def test_shifted_plans_repeat_exactly_and_never_overlap():
    tokens = torch.arange(211)
    before = torch.get_rng_state().clone()
    opts = dict(batch_size=3, block_size=5, eval_batches=3, seed=193)
    first = prepare_shifted_evaluation_index_plan(tokens, **opts)
    second = prepare_shifted_evaluation_index_plan(tokens, **opts)
    assert first.tokens is tokens
    assert all(torch.equal(a, b) for a, b in zip(first.starts, second.starts))
    assert all(
        torch.equal(x[:, 1:], y[:, :-1]) for x, y in first.batches()
    )
    metrics = analyze_index_plan_overlap(first)
    assert metrics.window_count == 9
    assert metrics.duplicate_start_count == 0
    assert metrics.repeated_window_token_positions == 0
    phases = {int(start) % 6 for start in torch.cat(first.starts)}
    assert len(phases) == 1
    assert torch.equal(before, torch.get_rng_state())


def test_multiple_seeds_explore_different_lattice_phases():
    tokens = torch.arange(131)
    phases = {
        int(plan.starts[0][0]) % 5
        for seed in range(20)
        for plan in [prepare_shifted_evaluation_index_plan(
            tokens, batch_size=2, block_size=4, eval_batches=2, seed=seed,
        )]
    }
    assert len(phases) > 1
    assert phases <= set(range(5))


def test_full_capacity_forces_phase_zero():
    assert feasible_lattice_phases(
        corpus_token_count=20, block_size=4, window_count=4,
    ) == 1
    plan = prepare_shifted_evaluation_index_plan(
        torch.arange(20), batch_size=2, block_size=4,
        eval_batches=2, seed=99,
    )
    assert sorted(torch.cat(plan.starts).tolist()) == [0, 5, 10, 15]


def test_feasible_phase_count_depends_on_budget():
    assert feasible_lattice_phases(
        corpus_token_count=22, block_size=4, window_count=4,
    ) == 3
    assert feasible_lattice_phases(
        corpus_token_count=22, block_size=4, window_count=3,
    ) == 5


def test_excess_capacity_rejects_before_rng_draw(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Cannot sample when capacity is too small")
    monkeypatch.setattr(torch, "randint", forbidden)
    with pytest.raises(ValueError, match="only 3"):
        prepare_shifted_evaluation_index_plan(
            torch.arange(19), batch_size=2, block_size=4,
            eval_batches=2,
        )


@pytest.mark.parametrize("name,bad", [
    ("batch_size", 0), ("batch_size", True),
    ("block_size", -1), ("eval_batches", 2.5),
    ("seed", -1), ("seed", False),
])
def test_invalid_configuration_is_rejected(name, bad):
    kwargs = dict(batch_size=2, block_size=3, eval_batches=2, seed=1)
    kwargs[name] = bad
    with pytest.raises(ValueError, match=name):
        prepare_shifted_evaluation_index_plan(torch.arange(50), **kwargs)


def test_invalid_corpus_dtype_rejected():
    with pytest.raises(ValueError, match="torch.long"):
        prepare_shifted_evaluation_index_plan(
            torch.arange(100, dtype=torch.int32), batch_size=2,
            block_size=4,
        )
