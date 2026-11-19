"""Population-size-weighted estimates for stratified validation *starts*.

Every valid start is an element of the target population. If starts
were uniformly sampled with replacement *within each region*, the
region-size-weighted mean is unbiased for a uniform-start mean of
the supplied per-window scores. It is not a confidence interval.
"""

from collections.abc import Sequence
from dataclasses import dataclass
import math

import torch

from rebuild_gpt.balanced_evaluation import start_region_bounds


@dataclass(frozen=True)
class StartUniformEstimate:
    sampled_window_count: int
    total_eligible_starts: int
    region_start_counts: tuple[int, ...]
    region_population_sizes: tuple[int, ...]
    region_mean_scores: tuple[float, ...]
    region_population_weights: tuple[float, ...]
    sampled_window_mean: float
    start_uniform_estimate: float

    @property
    def reweighting_difference(self) -> float:
        return self.start_uniform_estimate - self.sampled_window_mean


def estimate_start_uniform_score(
    starts: Sequence[int] | torch.Tensor,
    scores: Sequence[float],
    *,
    corpus_token_count: int,
    block_size: int,
    region_count: int,
) -> StartUniformEstimate:
    """Weight the *within-region* average by eligible start population.

    This estimator requires at least one observed score per region
    and exact one-to-one alignment between starts and per-window
    scalar scores. Do not substitute per-batch mean losses when one
    batch contains starts from different regions.
    """
    bounds = start_region_bounds(
        corpus_token_count=corpus_token_count,
        block_size=block_size, region_count=region_count,
    )
    if isinstance(starts, torch.Tensor):
        if starts.ndim != 1 or starts.dtype != torch.long or starts.device.type != "cpu":
            raise ValueError("starts must be flat CPU torch.long indices")
        positions = starts.tolist()
    elif isinstance(starts, Sequence) and not isinstance(starts, (str, bytes)):
        positions = list(starts)
    else:
        raise ValueError("starts must be a nonempty integer sequence")
    if not isinstance(scores, Sequence) or isinstance(scores, (str, bytes)):
        raise ValueError("scores must be a nonempty finite real sequence")
    values = tuple(scores)
    if not positions or len(positions) != len(values):
        raise ValueError("starts and scores must be nonempty and equally sized")
    if any(type(x) is not int or x < 0 or x >= corpus_token_count - block_size
           for x in positions):
        raise ValueError("starts contain invalid eligible window positions")
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) for v in values):
        raise ValueError("scores must contain finite real values")
    counts = []
    means = []
    sizes = []
    for low, high in bounds:
        selected = [v for start, v in zip(positions, values) if low <= start < high]
        if not selected:
            raise ValueError("every start region needs at least one score")
        counts.append(len(selected))
        sizes.append(high - low)
        means.append(sum(selected) / len(selected))
    population = corpus_token_count - block_size
    weights = tuple(size / population for size in sizes)
    estimate = sum(weight * mean for weight, mean in zip(weights, means))
    return StartUniformEstimate(
        sampled_window_count=len(values),
        total_eligible_starts=population,
        region_start_counts=tuple(counts),
        region_population_sizes=tuple(sizes),
        region_mean_scores=tuple(means),
        region_population_weights=weights,
        sampled_window_mean=sum(values) / len(values),
        start_uniform_estimate=estimate,
    )
