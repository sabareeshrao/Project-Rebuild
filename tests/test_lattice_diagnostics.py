"""Lattice reports reveal phase and capacity without redrawing windows."""

import pytest
import torch

from rebuild_gpt.disjoint_evaluation import prepare_disjoint_evaluation_index_plan
from rebuild_gpt.eval_index_plan import EvaluationIndexPlan
from rebuild_gpt.lattice_diagnostics import describe_disjoint_lattice
from rebuild_gpt.shifted_evaluation import prepare_shifted_evaluation_index_plan


def test_fixed_origin_plan_has_zero_phase():
    plan = prepare_disjoint_evaluation_index_plan(
        torch.arange(40), batch_size=2, block_size=4, eval_batches=2, seed=7,
    )
    report = describe_disjoint_lattice(plan)
    assert report.phase == 0
    assert report.tile_width == 5
    assert report.candidate_tiles_at_phase == 8
    assert report.selected_tiles == 4
    assert report.unselected_tiles == 4
    assert report.unused_prefix_tokens == 0
    assert report.unused_suffix_tokens == 0
    assert report.selected_corpus_coverage_fraction == pytest.approx(0.5)


def test_shifted_phase_and_unselected_tile_count_are_consistent():
    plan = prepare_shifted_evaluation_index_plan(
        torch.arange(53), batch_size=3, block_size=4,
        eval_batches=2, seed=194,
    )
    before = torch.get_rng_state().clone()
    report = describe_disjoint_lattice(plan)
    assert 0 <= report.phase <= 4
    assert report.phase == int(plan.starts[0][0]) % 5
    assert report.phase_count_feasible_for_budget == 5
    assert report.candidate_tiles_at_phase >= 6
    assert report.unselected_tiles == report.candidate_tiles_at_phase - 6
    assert report.unused_prefix_tokens == report.phase
    assert report.unused_prefix_tokens + report.unused_suffix_tokens <= 9
    assert torch.equal(before, torch.get_rng_state())


def test_exact_capacity_can_force_phase_zero():
    plan = prepare_shifted_evaluation_index_plan(
        torch.arange(20), batch_size=2, block_size=4,
        eval_batches=2, seed=2,
    )
    report = describe_disjoint_lattice(plan)
    assert report.phase_count_feasible_for_budget == 1
    assert report.phase == 0


def test_reject_mixed_phase_and_repeated_positions():
    tokens = torch.arange(40)
    bad_phase = EvaluationIndexPlan(
        tokens=tokens, starts=(torch.tensor([0, 6]),),
        batch_size=2, block_size=3, seed=1,
    )
    with pytest.raises(ValueError, match="phase"):
        describe_disjoint_lattice(bad_phase)
    repeated = EvaluationIndexPlan(
        tokens=tokens, starts=(torch.tensor([0, 0]),),
        batch_size=2, block_size=3, seed=1,
    )
    with pytest.raises(ValueError, match="nonoverlapping"):
        describe_disjoint_lattice(repeated)
