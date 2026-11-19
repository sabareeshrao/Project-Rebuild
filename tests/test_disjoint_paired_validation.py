"""Paired GPT comparison supports two explicit sampling distributions."""

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
    torch.manual_seed(190)
    return GPTLanguageModel(
        GPTConfig(7, 6, 8, 2, 1, attention_impl="sdpa", dropout=0.2)
    )


def test_disjoint_paired_windows_match_models_and_have_no_overlap():
    first = model().train()
    second = copy.deepcopy(first).eval()
    tokens = torch.arange(256) % 7
    before = torch.get_rng_state().clone()
    result = compare_paired_batches(
        first, second, tokens, seeds=(1, 2), batch_size=2,
        block_size=5, eval_batches=3, sampling_policy="disjoint",
    )
    assert result.sampling_policy == "disjoint"
    assert result.per_seed_means == (0.0, 0.0)
    assert len(result.overlap_by_seed) == 2
    assert all(item.window_count == 6 for item in result.overlap_by_seed)
    assert all(item.repeated_window_token_positions == 0
               for item in result.overlap_by_seed)
    assert first.training and not second.training
    assert torch.equal(before, torch.get_rng_state())


def test_replacement_default_uses_unchanged_paired_losses():
    first = model()
    second = copy.deepcopy(first)
    with torch.no_grad():
        second.head.linear.weight.add_(0.2 * torch.randn_like(second.head.linear.weight))
    tokens = torch.arange(175) % 7
    options = dict(seeds=(5, 6), batch_size=3, block_size=4, eval_batches=2)
    original = compare_paired_validation(first, second, tokens, **options)
    default = compare_paired_batches(first, second, tokens, **options)
    explicit = compare_paired_batches(
        first, second, tokens, sampling_policy="replacement", **options,
    )
    assert default.per_seed_means == original.candidate_minus_baseline
    assert default == explicit
    assert default.sampling_policy == "replacement"


def test_disjoint_capacity_rejects_instead_of_reusing_windows():
    first = model()
    with pytest.raises(ValueError, match="disjoint windows require"):
        compare_paired_batches(
            first, first, torch.arange(30) % 7,
            seeds=(1,), batch_size=3, block_size=5,
            eval_batches=2, sampling_policy="disjoint",
        )


@pytest.mark.parametrize("policy", ["uniform", "", None, 1])
def test_unknown_sampling_policy_rejected(policy):
    first = model()
    with pytest.raises(ValueError, match="sampling_policy"):
        compare_paired_batches(
            first, first, torch.arange(125) % 7,
            seeds=(1,), batch_size=2, block_size=5,
            sampling_policy=policy,
        )


def test_saved_checkpoint_comparison_respects_disjoint_policy(tmp_path):
    tokenizer = CharTokenizer("abcdefg")
    first = model().eval()
    path = tmp_path / "model.pt"
    save_text_checkpoint(first, tokenizer, path)
    corpus = tmp_path / "text.txt"
    corpus.write_text("abcdefg" * 150, encoding="utf-8")
    report = compare_saved_checkpoints(
        path, path, corpus, seeds=(1, 2), batch_size=2,
        block_size=4, eval_batches=2, sampling_policy="disjoint",
    )
    assert report.sampling_policy == "disjoint"
    assert report.paired.per_seed_means == (0.0, 0.0)
    assert all(x.repeated_window_token_positions == 0
               for x in report.paired.overlap_by_seed)
