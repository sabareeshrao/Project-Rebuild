"""Estimate cache-setup amortization from *measured* per-evaluation medians.

This is descriptive arithmetic, not a model of variance, scheduling,
data transfer overlap, or future workload performance.
"""

import math


def _check_timings(
    preparation_ms: float,
    regenerated_ms: float,
    cached_ms: float,
) -> None:
    for name, value in (
        ("preparation_ms", preparation_ms),
        ("regenerated_ms", regenerated_ms),
        ("cached_ms", cached_ms),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite nonnegative number")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")


def minimum_break_even_evaluations(
    *, preparation_ms: float, regenerated_ms: float, cached_ms: float,
) -> int | None:
    """Return min positive evaluations to amortize preparation, if possible.

    If caching is no faster per evaluation, no finite number of
    evaluations can recover an initial preparation overhead.
    """
    _check_timings(preparation_ms, regenerated_ms, cached_ms)
    savings_per_event = regenerated_ms - cached_ms
    if savings_per_event <= 0:
        return None
    return max(1, math.ceil(preparation_ms / savings_per_event))


def estimated_net_savings_ms(
    *, preparation_ms: float, regenerated_ms: float, cached_ms: float,
    evaluation_events: int,
) -> float:
    """Difference: E * regenerate - (preparation + E * cached)."""
    _check_timings(preparation_ms, regenerated_ms, cached_ms)
    if type(evaluation_events) is not int or evaluation_events < 0:
        raise ValueError("evaluation_events must be a nonnegative integer")
    return evaluation_events * (regenerated_ms - cached_ms) - preparation_ms
