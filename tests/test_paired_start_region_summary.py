"""Saved-model evaluations preserve balanced start diagnostics across seeds."""

import copy

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.checkpoint_comparison import compare_saved_checkpoints
from rebuild_gpt.paired_batch_evaluation import compare_paired_batches
from rebuild_gpt.text_checkpoint import save_text_checkpoint
from rebuild_gpt.tokenizer import CharTokenizer


def model():
    torch.manual_seed(220)
    return GPTLanguageModel(
        GPTConfig(7, 6, 8, 2, 1, dropout=0.2, attention_impl="sdpa")
    )


def test_paired_balanced_plans_report_pooled_and_per_seed_start_counts():
    baseline = model().train()
    candidate = copy.deepcopy(baseline).eval()
    rng = torch.get_rng_state().clone()
    result = compare_paired_batches(
        baseline, candidate, torch.arange(200) % 7,
        seeds=(1, 2, 3), batch_size=3, block_size=5,
        eval_batches=4, sampling_policy="balanced", sampling_regions=5,
    )
    summary = result.aggregate_start_regions
    assert summary is not None
    assert summary.seed_count == 3
    assert summary.window_count == result.cross_seed_overlap.window_count == 36
    assert summary.aggregate_counts == (9, 9, 6, 6, 6)
    assert summary.maximum_seed_imbalance == 1
    assert summary.aggregate_imbalance == 3
    assert summary.seeds_with_any_empty_region == 0
    assert summary.per_seed == result.start_regions_by_seed
    assert result.per_seed_means == (0.0, 0.0, 0.0)
    assert baseline.training and not candidate.training
    assert torch.equal(rng, torch.get_rng_state())


def test_replacement_does_not_add_implicit_region_requirements():
    baseline = model()
    # Exactly seven corpus tokens and T=5 leave only two valid starts.
    # The existing replacement sampler must not require four regions.
    candidate = copy.deepcopy(baseline)
    result = compare_paired_batches(
        baseline, candidate, torch.arange(7) % 7,
        seeds=(1,), batch_size=1, block_size=5,
        eval_batches=1,
    )
    assert result.aggregate_start_regions is None
    assert result.sampling_regions is None
    assert result.per_seed_means == (0.0,)


def test_saved_balanced_checkpoint_retains_pooled_summary(tmp_path):
    baseline = model().eval()
    checkpoint = tmp_path / "model.pt"
    save_text_checkpoint(baseline, CharTokenizer("abcdefg"), checkpoint)
    corpus = tmp_path / "text.txt"
    corpus.write_text("abcdefg" * 140, encoding="utf-8")
    result = compare_saved_checkpoints(
        checkpoint, checkpoint, corpus, seeds=(2, 3),
        batch_size=3, block_size=4, eval_batches=2,
        sampling_policy="balanced", sampling_regions=4,
    )
    summary = result.paired.aggregate_start_regions
    assert summary is not None
    assert summary.seed_count == 2
    assert summary.window_count == 12
    assert summary.aggregate_counts == (4, 4, 2, 2)
    assert summary.seeds_with_any_empty_region == 0
    assert result.paired.per_seed_means == (0.0, 0.0)
