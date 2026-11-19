"""Compare two GPTs on identical starts with region-size reweighting.

Scores are per-window mean next-token losses. Reweighting estimates
the population average over all valid starts, not a confidence bound.
"""

from dataclasses import dataclass

from rebuild_gpt.eval_index_plan import EvaluationIndexPlan
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.start_uniform_estimator import (
    StartUniformEstimate, estimate_start_uniform_score,
)
from rebuild_gpt.window_loss_evaluation import (
    WindowLossEvaluation, collect_window_losses,
)


@dataclass(frozen=True)
class StratifiedPairedComparison:
    baseline: StartUniformEstimate
    candidate: StartUniformEstimate
    candidate_minus_baseline: StartUniformEstimate
    baseline_windows: WindowLossEvaluation
    candidate_windows: WindowLossEvaluation

    @property
    def corrected_paired_delta(self) -> float:
        return self.candidate_minus_baseline.start_uniform_estimate

    @property
    def raw_paired_delta(self) -> float:
        return self.candidate_minus_baseline.sampled_window_mean


def compare_balanced_window_losses(
    baseline: GPTLanguageModel,
    candidate: GPTLanguageModel,
    plan: EvaluationIndexPlan,
    *,
    region_count: int = 4,
) -> StratifiedPairedComparison:
    """Evaluate both GPTs on the same windows, then reweight by region size.

    The plan must cover each eligible-start region. This method
    does not check the sampling mechanism; the unbiasedness
    argument needs *uniform random draws within each region*.
    """
    baseline_scores = collect_window_losses(baseline, plan)
    candidate_scores = collect_window_losses(candidate, plan)
    if baseline_scores.starts != candidate_scores.starts:
        raise RuntimeError("paired model comparisons require identical starts")
    starts = baseline_scores.starts
    kwargs = {
        "corpus_token_count": plan.tokens.numel(),
        "block_size": plan.block_size,
        "region_count": region_count,
    }
    a = estimate_start_uniform_score(
        starts, baseline_scores.per_window_losses, **kwargs,
    )
    b = estimate_start_uniform_score(
        starts, candidate_scores.per_window_losses, **kwargs,
    )
    delta = tuple(
        bv - av for av, bv in zip(
            baseline_scores.per_window_losses,
            candidate_scores.per_window_losses,
        )
    )
    differences = estimate_start_uniform_score(starts, delta, **kwargs)
    return StratifiedPairedComparison(
        baseline=a, candidate=b,
        candidate_minus_baseline=differences,
        baseline_windows=baseline_scores,
        candidate_windows=candidate_scores,
    )
