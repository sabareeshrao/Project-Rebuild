"""Stratified paired evaluation uses identical starts and correct signs."""

import copy

import pytest
import torch

from rebuild_gpt.balanced_evaluation import prepare_balanced_evaluation_index_plan
from rebuild_gpt.config import GPTConfig
from rebuild_gpt.eval_index_plan import prepare_evaluation_index_plan
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.stratified_model_comparison import compare_balanced_window_losses


def model():
    torch.manual_seed(226)
    return GPTLanguageModel(
        GPTConfig(7, 5, 8, 2, 1, dropout=0.2, attention_impl="sdpa")
    )


def plan():
    return prepare_balanced_evaluation_index_plan(
        torch.arange(150) % 7, batch_size=3, block_size=4,
        eval_batches=2, seed=226, region_count=4,
    )


def test_identical_models_produce_zero_raw_and_weighted_paired_deltas():
    first = model().train()
    second = copy.deepcopy(first).eval()
    state = torch.get_rng_state().clone()
    report = compare_balanced_window_losses(
        first, second, plan(), region_count=4,
    )
    assert report.corrected_paired_delta == 0
    assert report.raw_paired_delta == 0
    assert report.baseline_windows.starts == report.candidate_windows.starts
    assert len(report.baseline_windows.per_window_losses) == 6
    assert first.training and not second.training
    assert torch.equal(state, torch.get_rng_state())


def test_changed_weights_produce_paired_estimate_with_correct_region_accounting():
    first = model().eval()
    candidate = copy.deepcopy(first)
    with torch.no_grad():
        candidate.head.linear.weight[0, 0].add_(0.4)
    report = compare_balanced_window_losses(
        first, candidate, plan(), region_count=4,
    )
    assert report.baseline.region_start_counts == (2, 2, 1, 1)
    assert report.baseline.region_start_counts == report.candidate.region_start_counts
    assert report.corrected_paired_delta == pytest.approx(
        report.candidate.start_uniform_estimate
        - report.baseline.start_uniform_estimate,
        abs=1e-6,
    )
    assert report.raw_paired_delta == pytest.approx(
        report.candidate.sampled_window_mean
        - report.baseline.sampled_window_mean,
        abs=1e-6,
    )


def test_nonbalanced_missing_region_samples_are_rejected():
    first = model()
    raw = prepare_evaluation_index_plan(
        torch.arange(150) % 7, batch_size=2, block_size=4,
        eval_batches=1, seed=2,
    )
    with pytest.raises(ValueError, match="every start region"):
        compare_balanced_window_losses(
            first, first, raw, region_count=4,
        )


def test_insufficient_model_context_fails_before_parameter_update():
    first = model()
    bad = prepare_balanced_evaluation_index_plan(
        torch.arange(150) % 7, batch_size=2, block_size=7,
        eval_batches=2, seed=2, region_count=4,
    )
    with pytest.raises(ValueError, match="block_size"):
        compare_balanced_window_losses(first, first, bad)
