"""Descriptive direction counts for paired sampled validation differences.

These are counts of dependent sampled windows, NOT binomial trials or
a statistical significance test.
"""

from collections.abc import Sequence
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DirectionCounts:
    improved: int
    worsened: int
    tied: int

    @property
    def total(self) -> int:
        return self.improved + self.worsened + self.tied

    @property
    def improved_fraction(self) -> float:
        return self.improved / self.total


@dataclass(frozen=True)
class PairedDiagnostics:
    batch_directions: DirectionCounts
    seed_mean_directions: DirectionCounts
    per_seed_means: tuple[float, ...]
    mean_paired_delta: float
    tie_tolerance: float


def _direction_counts(values: Sequence[float], tolerance: float) -> DirectionCounts:
    return DirectionCounts(
        improved=sum(value < -tolerance for value in values),
        worsened=sum(value > tolerance for value in values),
        tied=sum(-tolerance <= value <= tolerance for value in values),
    )


def summarize_paired_deltas(
    per_seed_deltas: Sequence[Sequence[float]],
    *,
    tie_tolerance: float = 0.0,
) -> PairedDiagnostics:
    """Summarize signs without treating overlapping windows as independent.

    All seeds must have the same number of batches. A tolerance can
    label changes whose absolute magnitude is small as descriptive ties.
    """
    if (
        isinstance(tie_tolerance, bool)
        or not isinstance(tie_tolerance, (int, float))
        or not math.isfinite(tie_tolerance)
        or tie_tolerance < 0
    ):
        raise ValueError("tie_tolerance must be finite and nonnegative")
    if (not isinstance(per_seed_deltas, Sequence)
            or isinstance(per_seed_deltas, (str, bytes))
            or not per_seed_deltas):
        raise ValueError("per_seed_deltas must be nonempty, equally sized sequences")
    values = []
    expected_count = None
    for seed_losses in per_seed_deltas:
        if (
            not isinstance(seed_losses, Sequence)
            or isinstance(seed_losses, (str, bytes))
            or not seed_losses
        ):
            raise ValueError("per_seed_deltas must be nonempty, equally sized sequences")
        batch = tuple(seed_losses)
        if expected_count is None:
            expected_count = len(batch)
        elif len(batch) != expected_count:
            raise ValueError("per_seed_deltas must have equal batch counts")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in batch
        ):
            raise ValueError("per_seed_deltas must contain finite real numbers")
        values.append(batch)
    means = tuple(sum(batch) / len(batch) for batch in values)
    flat = tuple(value for batch in values for value in batch)
    return PairedDiagnostics(
        batch_directions=_direction_counts(flat, tie_tolerance),
        seed_mean_directions=_direction_counts(means, tie_tolerance),
        per_seed_means=means,
        mean_paired_delta=sum(means) / len(means),
        tie_tolerance=float(tie_tolerance),
    )
