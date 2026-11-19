"""Separate overlap within seeds from shared coverage across seed plans.

All plans must reference the same CPU token stream. Repeated token
positions count multiplicity, not independent statistical trials.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from rebuild_gpt.eval_index_plan import EvaluationIndexPlan
from rebuild_gpt.window_overlap import analyze_window_starts, analyze_index_plan_overlap


@dataclass(frozen=True)
class CrossSeedOverlap:
    seed_count: int
    window_count: int
    unique_start_count: int
    duplicate_starts_within_seeds: int
    duplicate_starts_between_seeds: int
    unique_corpus_token_positions: int
    repeated_positions_within_seeds: int
    repeated_positions_between_seeds: int
    total_window_token_positions: int
    corpus_token_count: int

    @property
    def coverage_fraction(self) -> float:
        return self.unique_corpus_token_positions / self.corpus_token_count

    @property
    def total_repeated_positions(self) -> int:
        return (
            self.repeated_positions_within_seeds
            + self.repeated_positions_between_seeds
        )


def analyze_cross_seed_overlap(
    plans: Sequence[EvaluationIndexPlan],
) -> CrossSeedOverlap:
    """Union all windows, distinguishing within-plan and between-plan reuse.

    'Between seeds' is the duplicate coverage remaining after removing
    repetition that already happened inside each seed's plan.
    """
    if (
        not isinstance(plans, Sequence)
        or isinstance(plans, (str, bytes))
        or not plans
    ):
        raise ValueError("plans must be a nonempty sequence of index plans")
    first = plans[0]
    if not isinstance(first, EvaluationIndexPlan):
        raise ValueError("plans must contain EvaluationIndexPlan instances")
    seeds = set()
    starts = []
    within_positions = 0
    within_duplicate_starts = 0
    within_distinct_coverage = 0
    for plan in plans:
        if not isinstance(plan, EvaluationIndexPlan):
            raise ValueError("plans must contain EvaluationIndexPlan instances")
        if plan.seed in seeds:
            raise ValueError("plans must have unique seeds")
        seeds.add(plan.seed)
        if (
            plan.block_size != first.block_size
            or plan.tokens.device.type != "cpu"
            or plan.tokens.dtype != torch.long
            or plan.tokens.numel() != first.tokens.numel()
            or plan.tokens.data_ptr() != first.tokens.data_ptr()
        ):
            raise ValueError("plans must share one CPU corpus and block_size")
        if not plan.starts:
            raise ValueError("plans must contain nonempty starts")
        per_seed = analyze_index_plan_overlap(plan)
        starts.extend(plan.starts)
        within_positions += per_seed.repeated_window_token_positions
        within_duplicate_starts += per_seed.duplicate_start_count
        within_distinct_coverage += per_seed.unique_corpus_token_positions
    union = analyze_window_starts(
        torch.cat(starts),
        block_size=first.block_size,
        corpus_token_count=first.tokens.numel(),
    )
    return CrossSeedOverlap(
        seed_count=len(plans),
        window_count=union.window_count,
        unique_start_count=union.unique_start_count,
        duplicate_starts_within_seeds=within_duplicate_starts,
        duplicate_starts_between_seeds=(
            union.duplicate_start_count - within_duplicate_starts
        ),
        unique_corpus_token_positions=union.unique_corpus_token_positions,
        repeated_positions_within_seeds=within_positions,
        repeated_positions_between_seeds=(
            within_distinct_coverage - union.unique_corpus_token_positions
        ),
        total_window_token_positions=union.total_window_token_positions,
        corpus_token_count=union.corpus_token_count,
    )
