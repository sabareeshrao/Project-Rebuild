"""Balanced starts can fairly pair two GPTs without changing the default."""

import copy

import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.paired_batch_evaluation import compare_paired_batches
from rebuild_gpt.paired_evaluation import compare_paired_validation
from rebuild_gpt.checkpoint_comparison import compare_saved_checkpoints
from rebuild_gpt.text_checkpoint import save_text_checkpoint
from rebuild_gpt.tokenizer import CharTokenizer


def model():
    torch.manual_seed(216)
    return GPTLanguageModel(
        GPTConfig(7, 6, 8, 2, 1, dropout=0.2, attention_impl="sdpa")
    )


def test_balanced_policy_pairs_identical_models_and_tracks_starts():
    original = model().train()
    equal = copy.deepcopy(original).eval()
    state = torch.get_rng_state().clone()
    result = compare_paired_batches(
        original, equal, torch.arange(230) % 7,
        seeds=(2, 3), batch_size=3, block_size=5,
        eval_batches=4, sampling_policy="balanced", sampling_regions=5,
    )
    assert result.per_seed_means == (0.0, 0.0)
    assert result.sampling_policy == "balanced"
    assert result.sampling_regions == 5
    assert result.lattice_by_seed == (None, None)
    assert all(item.empty_regions == 0 and item.imbalance <= 1
               for item in result.start_regions_by_seed)
    assert all(item.counts == (3, 3, 2, 2, 2)
               for item in result.start_regions_by_seed)
    assert result.cross_seed_overlap.window_count == 24
    assert original.training and not equal.training
    assert torch.equal(state, torch.get_rng_state())


def test_original_replacement_is_unchanged():
    original = model()
    candidate = copy.deepcopy(original)
    with torch.no_grad():
        candidate.head.linear.weight.add_(0.2 * torch.randn_like(candidate.head.linear.weight))
    opts = dict(seeds=(4, 5), batch_size=2, block_size=4, eval_batches=2)
    old = compare_paired_validation(
        original, candidate, torch.arange(220) % 7, **opts,
    )
    new = compare_paired_batches(
        original, candidate, torch.arange(220) % 7, **opts,
    )
    assert new.per_seed_means == old.candidate_minus_baseline
    assert new.sampling_regions is None
    assert new.start_regions_by_seed == (None, None)


@pytest.mark.parametrize("regions", [0, -1, True, 1.5])
def test_invalid_balanced_regions_rejected(regions):
    original = model()
    with pytest.raises(ValueError, match="sampling_regions"):
        compare_paired_batches(
            original, original, torch.arange(100) % 7,
            seeds=(1,), batch_size=2, block_size=4,
            sampling_policy="balanced", sampling_regions=regions,
        )


def test_bespoke_region_count_rejected_for_ignored_policy():
    original = model()
    with pytest.raises(ValueError, match="requires balanced"):
        compare_paired_batches(
            original, original, torch.arange(100) % 7,
            seeds=(1,), batch_size=2, block_size=4,
            sampling_policy="replacement", sampling_regions=5,
        )


def test_checkpoint_comparison_accepts_balanced_policy(tmp_path):
    original = model().eval()
    saved = tmp_path / "model.pt"
    save_text_checkpoint(original, CharTokenizer("abcdefg"), saved)
    text = tmp_path / "corpus.txt"
    text.write_text("abcdefg" * 160, encoding="utf-8")
    report = compare_saved_checkpoints(
        saved, saved, text, seeds=(1, 2),
        batch_size=3, block_size=4, eval_batches=2,
        sampling_policy="balanced", sampling_regions=5,
    )
    assert report.paired.per_seed_means == (0.0, 0.0)
    assert report.sampling_policy == "balanced"
    assert report.sampling_regions == 5
    assert all(item.empty_regions == 0
               for item in report.paired.start_regions_by_seed)
