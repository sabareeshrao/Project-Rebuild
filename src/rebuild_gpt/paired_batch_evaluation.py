"""Inspect matched per-batch validation losses for two GPT checkpoints.

The same seeded index plan is shared between models, so differences
reflect model weights rather than different sampled token windows.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from rebuild_gpt.eval_index_plan import prepare_evaluation_index_plan
from rebuild_gpt.disjoint_evaluation import prepare_disjoint_evaluation_index_plan
from rebuild_gpt.shifted_evaluation import prepare_shifted_evaluation_index_plan
from rebuild_gpt.balanced_evaluation import prepare_balanced_evaluation_index_plan
from rebuild_gpt.start_region_diagnostics import (
    StartRegionDiagnostics, analyze_start_regions,
)
from rebuild_gpt.lattice_diagnostics import LatticeDiagnostics, describe_disjoint_lattice
from rebuild_gpt.window_overlap import WindowOverlap, analyze_index_plan_overlap
from rebuild_gpt.cross_seed_overlap import CrossSeedOverlap, analyze_cross_seed_overlap
from rebuild_gpt.evaluation_samples import collect_split_losses
from rebuild_gpt.evaluation_stats import LossDispersion, summarize_losses
from rebuild_gpt.gpt import GPTLanguageModel


@dataclass(frozen=True)
class PairedBatchValidation:
    seeds: tuple[int, ...]
    baseline_losses: tuple[tuple[float, ...], ...]
    candidate_losses: tuple[tuple[float, ...], ...]
    differences: tuple[tuple[float, ...], ...]
    per_seed_means: tuple[float, ...]
    per_seed_summary: LossDispersion
    overlap_by_seed: tuple[WindowOverlap, ...]
    sampling_policy: str
    lattice_by_seed: tuple[LatticeDiagnostics | None, ...]
    cross_seed_overlap: CrossSeedOverlap
    start_regions_by_seed: tuple[StartRegionDiagnostics | None, ...]
    sampling_regions: int | None


def compare_paired_batches(
    baseline: GPTLanguageModel,
    candidate: GPTLanguageModel,
    tokens: torch.Tensor,
    *,
    seeds: Sequence[int],
    batch_size: int,
    block_size: int,
    eval_batches: int = 4,
    sampling_policy: str = "replacement",
    sampling_regions: int = 4,
) -> PairedBatchValidation:
    """Return the aligned per-batch candidate-minus-baseline losses."""
    if not isinstance(seeds, Sequence) or isinstance(seeds, (str, bytes)):
        raise ValueError("seeds must be unique nonnegative integers")
    chosen = tuple(seeds)
    if (
        not chosen
        or any(type(s) is not int or s < 0 for s in chosen)
        or len(set(chosen)) != len(chosen)
    ):
        raise ValueError("seeds must be unique nonnegative integers")
    if sampling_policy not in ("replacement", "disjoint", "shifted", "balanced"):
        raise ValueError("sampling_policy must be replacement, disjoint, shifted or balanced")
    if sampling_policy != "balanced" and sampling_regions != 4:
        raise ValueError("sampling_regions requires balanced sampling_policy")
    if sampling_policy == "balanced" and (
        type(sampling_regions) is not int or sampling_regions < 1
    ):
        raise ValueError("sampling_regions must be a positive integer")
    if not isinstance(tokens, torch.Tensor):
        raise ValueError("tokens must be a tensor")
    if baseline.config.vocab_size != candidate.config.vocab_size:
        raise ValueError("models must share vocabulary size")
    if (
        type(block_size) is not int or block_size < 1
        or block_size > min(baseline.config.block_size, candidate.config.block_size)
    ):
        raise ValueError("block_size exceeds a model context window")

    baseline_by_seed, candidate_by_seed, delta_by_seed, means = [], [], [], []
    overlaps = []
    lattices = []
    start_regions = []
    plans = []
    plan_builder = {
        "replacement": prepare_evaluation_index_plan,
        "disjoint": prepare_disjoint_evaluation_index_plan,
        "shifted": prepare_shifted_evaluation_index_plan,
        "balanced": prepare_balanced_evaluation_index_plan,
    }[sampling_policy]
    for seed in chosen:
        plan = plan_builder(
            tokens, batch_size=batch_size, block_size=block_size,
            eval_batches=eval_batches, seed=seed,
            **({"region_count": sampling_regions} if sampling_policy == "balanced" else {}),
        )
        plans.append(plan)
        overlaps.append(analyze_index_plan_overlap(plan))
        lattices.append(
            describe_disjoint_lattice(plan)
            if sampling_policy in ("disjoint", "shifted") else None
        )
        start_regions.append(
            analyze_start_regions(plan, region_count=sampling_regions)
            if sampling_policy == "balanced" else None
        )
        a = collect_split_losses(baseline, plan)
        b = collect_split_losses(candidate, plan)
        delta = tuple(y - x for x, y in zip(a, b))
        baseline_by_seed.append(a)
        candidate_by_seed.append(b)
        delta_by_seed.append(delta)
        means.append(sum(b) / len(b) - sum(a) / len(a))
    return PairedBatchValidation(
        seeds=chosen,
        baseline_losses=tuple(baseline_by_seed),
        candidate_losses=tuple(candidate_by_seed),
        differences=tuple(delta_by_seed),
        per_seed_means=tuple(means),
        per_seed_summary=summarize_losses(means),
        overlap_by_seed=tuple(overlaps),
        sampling_policy=sampling_policy,
        lattice_by_seed=tuple(lattices),
        cross_seed_overlap=analyze_cross_seed_overlap(plans),
        start_regions_by_seed=tuple(start_regions),
        sampling_regions=sampling_regions if sampling_policy == "balanced" else None,
    )
