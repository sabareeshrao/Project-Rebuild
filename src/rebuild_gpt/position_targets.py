"""Assess the first selected seed prefix that covers every positional bin.

The target is a retrospective experiment-design diagnostic, never
a statistical stopping rule or a model-performance guarantee.
"""

from collections.abc import Sequence
from dataclasses import dataclass
import math

from rebuild_gpt.position_growth import PositionGrowthStep


@dataclass(frozen=True)
class PositionTargetAssessment:
    target_fraction_per_bin: float
    required_positions_by_bin: tuple[int, ...]
    reached: bool
    first_reaching_seed_number: int | None
    first_reaching_seed: int | None
    final_positions_by_bin: tuple[int, ...]
    missing_positions_by_bin: tuple[int, ...]
    bins_below_target: int
    observed_seed_count: int


def assess_position_target(
    steps: Sequence[PositionGrowthStep],
    *,
    corpus_token_count: int,
    minimum_bin_fraction: float,
) -> PositionTargetAssessment:
    """Check when EVERY bin reaches its per-bin coverage threshold."""
    if type(corpus_token_count) is not int or corpus_token_count < 1:
        raise ValueError("corpus_token_count must be a positive integer")
    if (
        isinstance(minimum_bin_fraction, bool)
        or not isinstance(minimum_bin_fraction, (int, float))
        or not math.isfinite(minimum_bin_fraction)
        or not 0 < minimum_bin_fraction <= 1
    ):
        raise ValueError("minimum_bin_fraction must be finite and in (0, 1]")
    if (
        not isinstance(steps, Sequence)
        or isinstance(steps, (str, bytes))
        or not steps
        or any(not isinstance(step, PositionGrowthStep) for step in steps)
    ):
        raise ValueError("steps must be a nonempty sequence of PositionGrowthStep")
    count = len(steps[0].cumulative_positions_by_bin)
    if not 1 <= count <= corpus_token_count:
        raise ValueError("steps must have valid positional bin counts")
    sizes = tuple(
        (i + 1) * corpus_token_count // count
        - i * corpus_token_count // count
        for i in range(count)
    )
    required = tuple(math.ceil(size * minimum_bin_fraction) for size in sizes)
    previous = (0,) * count
    crossing = None
    for index, step in enumerate(steps, start=1):
        current = step.cumulative_positions_by_bin
        if (
            step.seed_number != index
            or len(current) != count
            or len(step.new_positions_by_bin) != count
            or any(
                type(now) is not int or not old <= now <= size
                or now - old != added
                for old, now, added, size in zip(
                    previous, current, step.new_positions_by_bin, sizes
                )
            )
            or sum(current) != step.cumulative_total_positions
        ):
            raise ValueError("steps must be an ordered, consistent coverage curve")
        if crossing is None and all(
            now >= goal for now, goal in zip(current, required)
        ):
            crossing = step
        previous = current
    missing = tuple(
        max(0, goal - now) for now, goal in zip(previous, required)
    )
    return PositionTargetAssessment(
        target_fraction_per_bin=float(minimum_bin_fraction),
        required_positions_by_bin=required,
        reached=crossing is not None,
        first_reaching_seed_number=(
            crossing.seed_number if crossing is not None else None
        ),
        first_reaching_seed=crossing.seed if crossing is not None else None,
        final_positions_by_bin=previous,
        missing_positions_by_bin=missing,
        bins_below_target=sum(value > 0 for value in missing),
        observed_seed_count=len(steps),
    )
