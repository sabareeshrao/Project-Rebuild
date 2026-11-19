"""Count sampled validation *starts* in corpus regions.

Start-position balance is distinct from token-position coverage:
contexts may cross boundaries and overlap.
"""

from dataclasses import dataclass

import torch

from rebuild_gpt.balanced_evaluation import start_region_bounds
from rebuild_gpt.eval_index_plan import EvaluationIndexPlan


@dataclass(frozen=True)
class StartRegionDiagnostics:
    region_count: int
    bounds: tuple[tuple[int, int], ...]
    counts: tuple[int, ...]
    sampled_window_count: int
    empty_regions: int
    minimum_count: int
    maximum_count: int
    imbalance: int


def analyze_start_regions(
    plan: EvaluationIndexPlan, *, region_count: int = 4,
) -> StartRegionDiagnostics:
    """Measure regional start counts using the same boundaries as balancing."""
    if not isinstance(plan, EvaluationIndexPlan) or not plan.starts:
        raise ValueError("plan must be a nonempty EvaluationIndexPlan")
    if (
        plan.tokens.device.type != "cpu" or plan.tokens.ndim != 1
        or plan.tokens.dtype != torch.long
    ):
        raise ValueError("plan tokens must be a flat CPU torch.long corpus")
    bounds = start_region_bounds(
        corpus_token_count=plan.tokens.numel(),
        block_size=plan.block_size, region_count=region_count,
    )
    for item in plan.starts:
        if (
            not isinstance(item, torch.Tensor) or item.ndim != 1
            or item.dtype != torch.long or item.device.type != "cpu"
            or item.numel() == 0
        ):
            raise ValueError("plan starts must be nonempty CPU int64 vectors")
    starts = torch.cat(plan.starts)
    if bool(torch.any(starts < 0)) or bool(torch.any(starts >= plan.tokens.numel() - plan.block_size)):
        raise ValueError("plan starts contain out-of-range positions")
    values = starts.tolist()
    counts = tuple(
        sum(low <= value < high for value in values)
        for low, high in bounds
    )
    assert sum(counts) == len(values)
    return StartRegionDiagnostics(
        region_count=region_count, bounds=bounds,
        counts=counts, sampled_window_count=len(values),
        empty_regions=sum(count == 0 for count in counts),
        minimum_count=min(counts), maximum_count=max(counts),
        imbalance=max(counts) - min(counts),
    )
