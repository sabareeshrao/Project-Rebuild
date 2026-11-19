"""Summarize eligible-start regional balance across multiple seed plans.

Regional start counts are not token coverage or effective sample sizes.
One seed may be badly concentrated even when pooled counts look even.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from rebuild_gpt.cross_seed_overlap import analyze_cross_seed_overlap
from rebuild_gpt.eval_index_plan import EvaluationIndexPlan
from rebuild_gpt.start_region_diagnostics import (
    StartRegionDiagnostics, analyze_start_regions,
)


@dataclass(frozen=True)
class StartRegionSummary:
    region_count: int
    seed_count: int
    window_count: int
    counts_by_seed: tuple[tuple[int, ...], ...]
    aggregate_counts: tuple[int, ...]
    empty_region_count_by_seed: tuple[int, ...]
    imbalance_by_seed: tuple[int, ...]
    seeds_with_any_empty_region: int
    aggregate_empty_regions: int
    aggregate_imbalance: int
    maximum_seed_imbalance: int
    per_seed: tuple[StartRegionDiagnostics, ...]


def summarize_start_regions(
    plans: Sequence[EvaluationIndexPlan], *, region_count: int = 4,
) -> StartRegionSummary:
    """Audit each seed and pooled start counts on an identical index partition.

    The full plan set is checked first to reject duplicate seeds or
    mismatched token streams. Aggregation does not redraw starts or
    advance global RNG.
    """
    coverage = analyze_cross_seed_overlap(plans)
    reports = tuple(
        analyze_start_regions(plan, region_count=region_count) for plan in plans
    )
    aggregate = tuple(
        sum(report.counts[i] for report in reports)
        for i in range(region_count)
    )
    if sum(aggregate) != coverage.window_count:
        raise RuntimeError("pooled start counts do not match sampled windows")
    empty = tuple(report.empty_regions for report in reports)
    imbalances = tuple(report.imbalance for report in reports)
    return StartRegionSummary(
        region_count=region_count,
        seed_count=len(reports),
        window_count=coverage.window_count,
        counts_by_seed=tuple(report.counts for report in reports),
        aggregate_counts=aggregate,
        empty_region_count_by_seed=empty,
        imbalance_by_seed=imbalances,
        seeds_with_any_empty_region=sum(count > 0 for count in empty),
        aggregate_empty_regions=sum(count == 0 for count in aggregate),
        aggregate_imbalance=max(aggregate) - min(aggregate),
        maximum_seed_imbalance=max(imbalances),
        per_seed=reports,
    )
