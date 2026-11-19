"""Cache economics are conditional on real timings, not promised improvements."""

import pytest

from rebuild_gpt.cache_economics import (
    estimated_net_savings_ms, minimum_break_even_evaluations,
)


def test_break_even_from_setup_and_repeated_savings():
    assert minimum_break_even_evaluations(
        preparation_ms=4, regenerated_ms=8, cached_ms=6,
    ) == 2
    assert minimum_break_even_evaluations(
        preparation_ms=4.1, regenerated_ms=8, cached_ms=6,
    ) == 3
    assert estimated_net_savings_ms(
        preparation_ms=4, regenerated_ms=8, cached_ms=6,
        evaluation_events=10,
    ) == pytest.approx(16)


def test_no_break_even_if_cached_evaluations_are_not_faster():
    for cached in (8, 9):
        assert minimum_break_even_evaluations(
            preparation_ms=3, regenerated_ms=8, cached_ms=cached,
        ) is None


def test_free_setup_still_needs_at_least_one_evaluation():
    assert minimum_break_even_evaluations(
        preparation_ms=0, regenerated_ms=2, cached_ms=1,
    ) == 1


@pytest.mark.parametrize("argument,bad", [
    ("preparation_ms", -1), ("regenerated_ms", True),
    ("cached_ms", float("nan")), ("cached_ms", float("inf")),
])
def test_invalid_timings_are_rejected(argument, bad):
    values = dict(preparation_ms=1, regenerated_ms=3, cached_ms=2)
    values[argument] = bad
    with pytest.raises(ValueError, match=argument):
        minimum_break_even_evaluations(**values)


@pytest.mark.parametrize("events", [-1, True, 2.5, "5"])
def test_invalid_event_count_is_rejected(events):
    with pytest.raises(ValueError, match="evaluation_events"):
        estimated_net_savings_ms(
            preparation_ms=1, regenerated_ms=3, cached_ms=2,
            evaluation_events=events,
        )
