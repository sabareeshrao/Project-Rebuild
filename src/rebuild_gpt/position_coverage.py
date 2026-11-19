"""Measure where sampled validation windows touch a held-out corpus.

This counts token positions (including the next-token target position)
rather than model loss or the statistical independence of samples.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from rebuild_gpt.cross_seed_overlap import analyze_cross_seed_overlap
from rebuild_gpt.eval_index_plan import EvaluationIndexPlan


@dataclass(frozen=True)
class PositionBin:
    index: int
    start: int
    stop: int
    unique_positions: int
    window_position_visits: int
    repeated_position_visits: int
    coverage_fraction: float


@dataclass(frozen=True)
class PositionCoverage:
    bins: tuple[PositionBin, ...]
    corpus_token_count: int
    window_count: int
    global_unique_positions: int
    zero_coverage_bins: int

    @property
    def global_coverage_fraction(self) -> float:
        return self.global_unique_positions / self.corpus_token_count


def analyze_position_coverage(
    plans: Sequence[EvaluationIndexPlan],
    *,
    position_bins: int = 8,
) -> PositionCoverage:
    """Partition the corpus into equal-proportion bins and count coverage.

    Bin boundaries are floor(i*N/B), so a corpus with N tokens
    has exactly B nonempty bins of nearly equal size when B <= N.
    Token intervals are half-open [start, start + block_size + 1).
    """
    union = analyze_cross_seed_overlap(plans)
    n = union.corpus_token_count
    if type(position_bins) is not int or not 1 <= position_bins <= min(n, 256):
        raise ValueError("position_bins must be 1..min(corpus tokens, 256)")

    intervals = sorted(
        (int(start), int(start) + plan.block_size + 1)
        for plan in plans
        for batch in plan.starts
        for start in batch
    )
    bins = []
    for i in range(position_bins):
        left, right = i * n // position_bins, (i + 1) * n // position_bins
        seen_up_to = left
        visited = 0
        unique = 0
        for start, stop in intervals:
            lo, hi = max(left, start), min(right, stop)
            if lo >= hi:
                continue
            visited += hi - lo
            unique += max(0, hi - max(lo, seen_up_to))
            seen_up_to = max(seen_up_to, hi)
        bins.append(PositionBin(
            index=i, start=left, stop=right,
            unique_positions=unique,
            window_position_visits=visited,
            repeated_position_visits=visited - unique,
            coverage_fraction=unique / (right - left),
        ))
    assert sum(b.unique_positions for b in bins) == union.unique_corpus_token_positions
    assert sum(b.window_position_visits for b in bins) == union.total_window_token_positions
    return PositionCoverage(
        bins=tuple(bins),
        corpus_token_count=n,
        window_count=union.window_count,
        global_unique_positions=union.unique_corpus_token_positions,
        zero_coverage_bins=sum(b.unique_positions == 0 for b in bins),
    )
