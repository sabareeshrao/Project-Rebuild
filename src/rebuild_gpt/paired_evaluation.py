"""Evaluate two GPT weights on exactly the same sampled validation windows."""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from rebuild_gpt.eval_index_plan import prepare_evaluation_index_plan
from rebuild_gpt.evaluation_samples import collect_split_losses
from rebuild_gpt.evaluation_stats import LossDispersion, summarize_losses
from rebuild_gpt.gpt import GPTLanguageModel


@dataclass(frozen=True)
class PairedValidation:
    seeds: tuple[int, ...]
    baseline_means: tuple[float, ...]
    candidate_means: tuple[float, ...]
    candidate_minus_baseline: tuple[float, ...]
    delta_summary: LossDispersion


def compare_paired_validation(
    baseline: GPTLanguageModel,
    candidate: GPTLanguageModel,
    tokens: torch.Tensor,
    *,
    seeds: Sequence[int],
    batch_size: int,
    block_size: int,
    eval_batches: int = 4,
) -> PairedValidation:
    """Compare models by paired mean losses over the same index plans.

    A negative candidate-minus-baseline loss is an improvement on
    these sampled windows; it is not a statistical significance test.
    """
    if not isinstance(seeds, Sequence) or isinstance(seeds, (str, bytes)):
        raise ValueError("seeds must be unique nonnegative integers")
    chosen = tuple(seeds)
    if (not chosen or any(type(s) is not int or s < 0 for s in chosen)
            or len(set(chosen)) != len(chosen)):
        raise ValueError("seeds must be unique nonnegative integers")
    if not isinstance(tokens, torch.Tensor):
        raise ValueError("tokens must be a torch.Tensor")
    if baseline.config.vocab_size != candidate.config.vocab_size:
        raise ValueError("models must share vocabulary size")
    if (type(block_size) is not int or block_size < 1
            or block_size > min(baseline.config.block_size, candidate.config.block_size)):
        raise ValueError("block_size exceeds a model context window")
    reference, alternate = [], []
    for seed in chosen:
        plan = prepare_evaluation_index_plan(
            tokens, batch_size=batch_size, block_size=block_size,
            eval_batches=eval_batches, seed=seed,
        )
        first = collect_split_losses(baseline, plan)
        second = collect_split_losses(candidate, plan)
        reference.append(sum(first) / len(first))
        alternate.append(sum(second) / len(second))
    a, b = tuple(reference), tuple(alternate)
    differences = tuple(candidate_loss - base_loss for base_loss, candidate_loss in zip(a, b))
    return PairedValidation(
        seeds=chosen, baseline_means=a, candidate_means=b,
        candidate_minus_baseline=differences,
        delta_summary=summarize_losses(differences),
    )
