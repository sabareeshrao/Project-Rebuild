"""Descriptive summaries for repeated sampled-evaluation losses.

Standard error is descriptive and assumes independent observations if
interpreted inferentially. Overlapping token windows violate that
assumption, so it must not be presented as a calibrated confidence bound.
"""

from dataclasses import dataclass
import math
from statistics import stdev
from collections.abc import Sequence


@dataclass(frozen=True)
class LossDispersion:
    count: int
    mean: float
    sample_stdev: float | None
    standard_error: float | None
    minimum: float
    maximum: float


def summarize_losses(losses: Sequence[float]) -> LossDispersion:
    """Summarize finite losses; undefined sample variation is None for n=1."""
    if not isinstance(losses, Sequence) or isinstance(losses, (str, bytes)):
        raise ValueError("losses must be a nonempty sequence of finite numbers")
    values = tuple(losses)
    if not values:
        raise ValueError("losses must be a nonempty sequence of finite numbers")
    for value in values:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            raise ValueError("losses must contain finite real numbers")
    mean = sum(values) / len(values)
    spread = stdev(values) if len(values) >= 2 else None
    return LossDispersion(
        count=len(values), mean=mean,
        sample_stdev=spread,
        standard_error=spread / math.sqrt(len(values)) if spread is not None else None,
        minimum=float(min(values)), maximum=float(max(values)),
    )
