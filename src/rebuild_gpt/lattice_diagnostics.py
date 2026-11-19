"""Inspect a fixed-origin or phase-shifted disjoint evaluation lattice.

A lattice contains full T+1-token tiles. Diagnostics describe
the chosen phase and eligible tile positions, not statistical
independence or estimator bias.
"""

from dataclasses import dataclass

import torch

from rebuild_gpt.eval_index_plan import EvaluationIndexPlan
from rebuild_gpt.shifted_evaluation import feasible_lattice_phases
from rebuild_gpt.window_overlap import analyze_index_plan_overlap


@dataclass(frozen=True)
class LatticeDiagnostics:
    phase: int
    phase_count_feasible_for_budget: int
    tile_width: int
    candidate_tiles_at_phase: int
    selected_tiles: int
    unselected_tiles: int
    unused_prefix_tokens: int
    unused_suffix_tokens: int
    selected_corpus_coverage_fraction: float


def describe_disjoint_lattice(plan: EvaluationIndexPlan) -> LatticeDiagnostics:
    """Recover and validate the lattice phase from selected starts.

    Reject mixed-phase or overlapping plans rather than assigning a
    misleading phase to arbitrary random-start samples.
    """
    if not isinstance(plan, EvaluationIndexPlan) or not plan.starts:
        raise ValueError("plan must be a nonempty EvaluationIndexPlan")
    starts = torch.cat(plan.starts)
    width = plan.block_size + 1
    if starts.numel() < 1:
        raise ValueError("plan must contain nonempty start positions")
    phase = int(starts[0]) % width
    if not bool(torch.all(starts % width == phase)):
        raise ValueError("plan starts are not on one disjoint lattice phase")
    overlap = analyze_index_plan_overlap(plan)
    if overlap.repeated_window_token_positions != 0:
        raise ValueError("plan starts must form nonoverlapping windows")
    total = plan.tokens.numel()
    candidates = (total - phase) // width
    count = starts.numel()
    return LatticeDiagnostics(
        phase=phase,
        phase_count_feasible_for_budget=feasible_lattice_phases(
            corpus_token_count=total, block_size=plan.block_size,
            window_count=count,
        ),
        tile_width=width,
        candidate_tiles_at_phase=candidates,
        selected_tiles=count,
        unselected_tiles=candidates - count,
        unused_prefix_tokens=phase,
        unused_suffix_tokens=total - phase - candidates * width,
        selected_corpus_coverage_fraction=count * width / total,
    )
