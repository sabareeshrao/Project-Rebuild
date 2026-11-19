"""Describe repeated and overlapping token windows in a sampled plan.

Each next-token window uses T+1 contiguous token positions. These
metrics are deterministic counting tools, not independence tests.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from rebuild_gpt.eval_index_plan import EvaluationIndexPlan


@dataclass(frozen=True)
class WindowOverlap:
    window_count: int
    unique_start_count: int
    duplicate_start_count: int
    overlapping_neighbor_pairs: int
    total_window_token_positions: int
    unique_corpus_token_positions: int
    repeated_window_token_positions: int
    corpus_token_count: int

    @property
    def coverage_fraction(self) -> float:
        return self.unique_corpus_token_positions / self.corpus_token_count


def analyze_window_starts(
    starts: Sequence[int] | torch.Tensor,
    *,
    block_size: int,
    corpus_token_count: int,
) -> WindowOverlap:
    """Count covered positions and overlap of sorted neighboring windows.

    The neighbor-pair count is NOT the number of every overlapping pair.
    It counts adjacent overlapping intervals after sorting by start.
    """
    if type(block_size) is not int or block_size < 1:
        raise ValueError("block_size must be positive")
    if type(corpus_token_count) is not int or corpus_token_count <= block_size:
        raise ValueError("corpus_token_count must exceed block_size")
    if isinstance(starts, torch.Tensor):
        if starts.ndim != 1 or starts.dtype != torch.long or starts.device.type != "cpu":
            raise ValueError("starts must be a one-dimensional CPU int64 sequence")
        values = starts.tolist()
    elif isinstance(starts, Sequence) and not isinstance(starts, (str, bytes)):
        values = list(starts)
    else:
        raise ValueError("starts must be a nonempty integer sequence")
    if not values or any(
        type(v) is not int or v < 0 or v >= corpus_token_count - block_size
        for v in values
    ):
        raise ValueError("starts must contain valid nonnegative window positions")

    ordered = sorted(values)
    end = -1
    unique_positions = 0
    neighbor_overlap = 0
    span = block_size + 1
    for i, start in enumerate(ordered):
        if i and start < ordered[i - 1] + span:
            neighbor_overlap += 1
        stop = start + span
        unique_positions += max(0, stop - max(end, start))
        end = max(end, stop)
    total = len(ordered) * span
    return WindowOverlap(
        window_count=len(ordered),
        unique_start_count=len(set(ordered)),
        duplicate_start_count=len(ordered) - len(set(ordered)),
        overlapping_neighbor_pairs=neighbor_overlap,
        total_window_token_positions=total,
        unique_corpus_token_positions=unique_positions,
        repeated_window_token_positions=total - unique_positions,
        corpus_token_count=corpus_token_count,
    )


def analyze_index_plan_overlap(plan: EvaluationIndexPlan) -> WindowOverlap:
    """Summarize all sampled batches in one seed's evaluation index plan."""
    if not isinstance(plan, EvaluationIndexPlan):
        raise ValueError("plan must be an EvaluationIndexPlan")
    starts = torch.cat(plan.starts)
    return analyze_window_starts(
        starts, block_size=plan.block_size,
        corpus_token_count=plan.tokens.numel(),
    )
