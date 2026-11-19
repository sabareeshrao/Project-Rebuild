"""Paired GPT reports capture cross-seed reuse even for disjoint plans."""

import copy

import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.paired_batch_evaluation import compare_paired_batches
from rebuild_gpt.paired_evaluation import compare_paired_validation


def model():
    torch.manual_seed(200)
    return GPTLanguageModel(
        GPTConfig(7, 5, 8, 2, 1, dropout=0.2, attention_impl="sdpa")
    )


@pytest.mark.parametrize("policy", ("disjoint", "shifted"))
def test_full_tile_coverage_per_seed_repeats_across_seeds(policy):
    first = model().train()
    second = copy.deepcopy(first).eval()
    rng = torch.get_rng_state().clone()
    # Twenty tokens are exactly four T+1 tiles. Both seeds must sample
    # the same four tiles, though their permutation may differ.
    result = compare_paired_batches(
        first, second, torch.arange(20) % 7,
        seeds=(1, 2), batch_size=2, block_size=4,
        eval_batches=2, sampling_policy=policy,
    )
    assert result.per_seed_means == (0.0, 0.0)
    assert all(x.repeated_window_token_positions == 0 for x in result.overlap_by_seed)
    coverage = result.cross_seed_overlap
    assert coverage.window_count == 8
    assert coverage.unique_start_count == 4
    assert coverage.duplicate_starts_within_seeds == 0
    assert coverage.duplicate_starts_between_seeds == 4
    assert coverage.unique_corpus_token_positions == 20
    assert coverage.repeated_positions_within_seeds == 0
    assert coverage.repeated_positions_between_seeds == 20
    assert coverage.coverage_fraction == 1
    assert first.training and not second.training
    assert torch.equal(rng, torch.get_rng_state())


def test_replacement_keeps_reference_loss_sequence_and_reports_union():
    first = model()
    second = copy.deepcopy(first)
    with torch.no_grad():
        second.head.linear.weight.add_(0.1 * torch.randn_like(second.head.linear.weight))
    tokens = torch.arange(200) % 7
    opts = dict(seeds=(4, 5, 6), batch_size=3, block_size=4, eval_batches=2)
    observed = compare_paired_batches(first, second, tokens, **opts)
    reference = compare_paired_validation(first, second, tokens, **opts)
    assert observed.per_seed_means == reference.candidate_minus_baseline
    assert observed.cross_seed_overlap.seed_count == 3
    assert observed.cross_seed_overlap.window_count == 18
    assert observed.cross_seed_overlap.total_repeated_positions == (
        observed.cross_seed_overlap.total_window_token_positions
        - observed.cross_seed_overlap.unique_corpus_token_positions
    )
    assert observed.cross_seed_overlap.repeated_positions_between_seeds >= 0


def test_one_seed_has_no_cross_seed_overlap():
    first = model()
    result = compare_paired_batches(
        first, first, torch.arange(80) % 7,
        seeds=(3,), batch_size=2, block_size=4, eval_batches=2,
    )
    assert result.cross_seed_overlap.seed_count == 1
    assert result.cross_seed_overlap.repeated_positions_between_seeds == 0
    assert result.cross_seed_overlap.duplicate_starts_between_seeds == 0
