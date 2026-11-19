"""Audit whether balanced window *starts* also cover corpus-token regions.

Start-region and token-position partitions are intentionally kept
separate. This is a reproducible coverage check, not a statistical
test of a validation estimator.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from rebuild_gpt.cross_seed_overlap import CrossSeedOverlap, analyze_cross_seed_overlap
from rebuild_gpt.eval_index_plan import EvaluationIndexPlan
from rebuild_gpt.position_coverage import PositionCoverage, analyze_position_coverage
from rebuild_gpt.start_region_summary import StartRegionSummary, summarize_start_regions


@dataclass(frozen=True)
class SamplingPolicyAudit:
    start_regions: StartRegionSummary
    token_positions: PositionCoverage
    cross_seed_overlap: CrossSeedOverlap
    all_seeds_touch_every_start_region: bool
    any_unsampled_token_bin: bool
    start_balance_without_full_token_bin_coverage: bool


def audit_start_and_token_coverage(
    plans: Sequence[EvaluationIndexPlan],
    *,
    start_regions: int = 4,
    token_bins: int = 8,
) -> SamplingPolicyAudit:
    """Check start balance and actual touched token bins from one plan set.

    If all seeds have starts in every region while some token bins
    remain untouched, flag the mismatch explicitly. Start regions
    partition eligible *start indices*, while token bins partition
    *all corpus token positions*. Neither implies unbiased model loss.
    """
    starts = summarize_start_regions(plans, region_count=start_regions)
    positions = analyze_position_coverage(plans, position_bins=token_bins)
    cross_seed = analyze_cross_seed_overlap(plans)
    if (
        starts.window_count != cross_seed.window_count
        or positions.global_unique_positions != cross_seed.unique_corpus_token_positions
    ):
        raise RuntimeError("start and token coverage accounting disagree")
    all_start_regions_visited = starts.seeds_with_any_empty_region == 0
    any_empty_token_bin = positions.zero_coverage_bins > 0
    return SamplingPolicyAudit(
        start_regions=starts,
        token_positions=positions,
        cross_seed_overlap=cross_seed,
        all_seeds_touch_every_start_region=all_start_regions_visited,
        any_unsampled_token_bin=any_empty_token_bin,
        start_balance_without_full_token_bin_coverage=(
            all_start_regions_visited and any_empty_token_bin
        ),
    )
