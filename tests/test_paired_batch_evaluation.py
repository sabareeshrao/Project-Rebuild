"""Matched evaluation batches preserve exact per-seed mean comparisons."""

import copy

import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.paired_batch_evaluation import compare_paired_batches
from rebuild_gpt.paired_evaluation import compare_paired_validation


def model():
    torch.manual_seed(183)
    return GPTLanguageModel(
        GPTConfig(7, 5, 8, 2, 1, dropout=0.2, attention_impl="sdpa")
    )


def test_identical_models_produce_zero_per_batch_deltas_and_keep_modes():
    original = model().train()
    same = copy.deepcopy(original).eval()
    before = torch.get_rng_state().clone()
    tokens = torch.arange(100) % 7
    result = compare_paired_batches(
        original, same, tokens, seeds=(4, 5),
        batch_size=2, block_size=4, eval_batches=3,
    )
    assert result.differences == ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    assert result.per_seed_means == (0.0, 0.0)
    assert original.training and not same.training
    assert torch.equal(before, torch.get_rng_state())


def test_batch_deltas_reconstruct_seed_means_from_existing_comparison():
    first = model()
    second = copy.deepcopy(first)
    with torch.no_grad():
        second.head.linear.weight.add_(0.1 * torch.randn_like(second.head.linear.weight))
    tokens = torch.arange(100) % 7
    params = dict(seeds=(1, 3, 5), batch_size=3, block_size=4, eval_batches=2)
    result = compare_paired_batches(first, second, tokens, **params)
    old = compare_paired_validation(first, second, tokens, **params)
    assert result.per_seed_means == old.candidate_minus_baseline
    for a, b, differences, mean in zip(
        result.baseline_losses, result.candidate_losses,
        result.differences, result.per_seed_means,
    ):
        assert differences == tuple(y - x for x, y in zip(a, b))
        assert mean == sum(b) / len(b) - sum(a) / len(a)
    assert result.per_seed_summary == old.delta_summary


def test_one_seed_and_one_batch_have_undefined_sample_spread():
    first = model()
    result = compare_paired_batches(
        first, copy.deepcopy(first), torch.arange(100) % 7,
        seeds=(1,), batch_size=1, block_size=3, eval_batches=1,
    )
    assert result.per_seed_summary.standard_error is None


@pytest.mark.parametrize("seeds", [[], (-1,), (1, 1), "1,2", (True,)])
def test_invalid_seed_sets_rejected(seeds):
    first = model()
    with pytest.raises(ValueError, match="seeds"):
        compare_paired_batches(
            first, first, torch.arange(100) % 7,
            seeds=seeds, batch_size=2, block_size=3,
        )


def test_block_size_exceeding_either_model_is_rejected():
    first = model()
    with pytest.raises(ValueError, match="block_size"):
        compare_paired_batches(
            first, copy.deepcopy(first), torch.arange(100) % 7,
            seeds=(1,), batch_size=2, block_size=6,
        )
