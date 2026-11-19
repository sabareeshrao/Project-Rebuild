"""Track which parts of a held-out corpus gain coverage with each seed."""

from collections.abc import Sequence
from dataclasses import dataclass

from rebuild_gpt.eval_index_plan import EvaluationIndexPlan
from rebuild_gpt.position_coverage import analyze_position_coverage


@dataclass(frozen=True)
class PositionGrowthStep:
    seed: int
    seed_number: int
    new_positions_by_bin: tuple[int, ...]
    cumulative_positions_by_bin: tuple[int, ...]
    newly_touched_bins: int
    new_total_positions: int
    cumulative_total_positions: int


def analyze_position_growth(
    plans: Sequence[EvaluationIndexPlan],
    *,
    position_bins: int = 8,
) -> tuple[PositionGrowthStep, ...]:
    """Calculate per-bin marginal union gains for an ordered seed sequence.

    Each prefix uses the same fixed bin boundaries and the existing
    exact interval-union calculation, without drawing new starts.
    """
    # Full-set validation happens first to reject incompatible
    # corpora or duplicate seeds before calculating a partial result.
    overall = analyze_position_coverage(plans, position_bins=position_bins)
    previous = (0,) * len(overall.bins)
    steps = []
    for index, plan in enumerate(plans, start=1):
        profile = analyze_position_coverage(plans[:index], position_bins=position_bins)
        current = tuple(b.unique_positions for b in profile.bins)
        added = tuple(now - old for old, now in zip(previous, current))
        if any(value < 0 for value in added):
            raise RuntimeError("per-bin corpus union decreased")
        steps.append(PositionGrowthStep(
            seed=plan.seed,
            seed_number=index,
            new_positions_by_bin=added,
            cumulative_positions_by_bin=current,
            newly_touched_bins=sum(
                old == 0 and now > 0 for old, now in zip(previous, current)
            ),
            new_total_positions=sum(added),
            cumulative_total_positions=sum(current),
        ))
        previous = current
    assert sum(steps[-1].cumulative_positions_by_bin) == overall.global_unique_positions
    return tuple(steps)
