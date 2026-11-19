"""Measure the incremental held-out token coverage added by each seed.

This is descriptive corpus coverage, not an effective sample size or
an inference about the independence of evaluation observations.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from rebuild_gpt.cross_seed_overlap import analyze_cross_seed_overlap
from rebuild_gpt.eval_index_plan import EvaluationIndexPlan
from rebuild_gpt.window_overlap import analyze_index_plan_overlap


@dataclass(frozen=True)
class CoverageGain:
    seed: int
    seed_number: int
    window_count: int
    positions_touched_this_seed: int
    new_corpus_token_positions: int
    already_covered_positions_this_seed: int
    repeated_positions_within_seed: int
    cumulative_unique_positions: int
    cumulative_coverage_fraction: float


def analyze_coverage_growth(
    plans: Sequence[EvaluationIndexPlan],
) -> tuple[CoverageGain, ...]:
    """Report each seed's *marginal* unique positions in supplied order.

    Each prefix is checked with the existing cross-seed analyzer.
    Prefix processing trades some extra CPU work for consistency with
    the authoritative interval-union accounting.
    """
    # Validate all plans and unique seed/corpus identity up front,
    # rather than reporting a partial curve before discovering errors.
    analyze_cross_seed_overlap(plans)
    steps = []
    previous_unique = 0
    for index, plan in enumerate(plans, start=1):
        prefix = analyze_cross_seed_overlap(plans[:index])
        local = analyze_index_plan_overlap(plan)
        added = prefix.unique_corpus_token_positions - previous_unique
        reused = local.unique_corpus_token_positions - added
        if added < 0 or reused < 0:
            raise RuntimeError("invalid coverage accounting: unique union decreased")
        steps.append(CoverageGain(
            seed=plan.seed,
            seed_number=index,
            window_count=local.window_count,
            positions_touched_this_seed=local.unique_corpus_token_positions,
            new_corpus_token_positions=added,
            already_covered_positions_this_seed=reused,
            repeated_positions_within_seed=local.repeated_window_token_positions,
            cumulative_unique_positions=prefix.unique_corpus_token_positions,
            cumulative_coverage_fraction=prefix.coverage_fraction,
        ))
        previous_unique = prefix.unique_corpus_token_positions
    return tuple(steps)
