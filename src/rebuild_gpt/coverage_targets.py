"""Describe when a preselected seed sequence first reaches a coverage goal.

A target is an experiment-design diagnostic, not an adaptive
statistical stopping rule and not a guarantee of model quality.
"""

from collections.abc import Sequence
from dataclasses import dataclass
import math

from rebuild_gpt.coverage_growth import CoverageGain


@dataclass(frozen=True)
class CoverageTargetAssessment:
    target_fraction: float
    target_token_positions: int
    corpus_token_count: int
    observed_seed_count: int
    reached: bool
    first_reaching_seed_number: int | None
    first_reaching_seed: int | None
    final_unique_positions: int
    additional_positions_needed: int


def assess_coverage_target(
    steps: Sequence[CoverageGain],
    *,
    corpus_token_count: int,
    target_fraction: float,
) -> CoverageTargetAssessment:
    """Find first prefix meeting a target among already-sampled seeds.

    The target is rounded *up* to an integer token count. The entire
    supplied sequence remains fixed; this helper never generates
    additional seeds or changes evaluation samples.
    """
    if type(corpus_token_count) is not int or corpus_token_count < 1:
        raise ValueError("corpus_token_count must be a positive integer")
    if (
        isinstance(target_fraction, bool)
        or not isinstance(target_fraction, (int, float))
        or not math.isfinite(target_fraction)
        or not 0 < target_fraction <= 1
    ):
        raise ValueError("target_fraction must be finite and in (0, 1]")
    if (
        not isinstance(steps, Sequence)
        or isinstance(steps, (str, bytes))
        or not steps
        or any(not isinstance(step, CoverageGain) for step in steps)
    ):
        raise ValueError("steps must be a nonempty sequence of CoverageGain")
    last_unique = 0
    for index, step in enumerate(steps, start=1):
        if (
            step.seed_number != index
            or step.new_corpus_token_positions < 0
            or step.cumulative_unique_positions != (
                last_unique + step.new_corpus_token_positions
            )
            or not 0 <= step.cumulative_unique_positions <= corpus_token_count
        ):
            raise ValueError("steps must be a consistent ordered coverage curve")
        last_unique = step.cumulative_unique_positions
    required = math.ceil(float(target_fraction) * corpus_token_count)
    reached_step = next(
        (step for step in steps if step.cumulative_unique_positions >= required),
        None,
    )
    return CoverageTargetAssessment(
        target_fraction=float(target_fraction),
        target_token_positions=required,
        corpus_token_count=corpus_token_count,
        observed_seed_count=len(steps),
        reached=reached_step is not None,
        first_reaching_seed_number=(
            reached_step.seed_number if reached_step is not None else None
        ),
        first_reaching_seed=reached_step.seed if reached_step is not None else None,
        final_unique_positions=last_unique,
        additional_positions_needed=max(0, required - last_unique),
    )
