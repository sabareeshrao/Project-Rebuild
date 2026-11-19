"""Paired GPT comparisons can use seeded phase-shifted disjoint starts."""

import copy

import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.checkpoint_comparison import compare_saved_checkpoints
from rebuild_gpt.paired_batch_evaluation import compare_paired_batches
from rebuild_gpt.text_checkpoint import save_text_checkpoint
from rebuild_gpt.tokenizer import CharTokenizer


def model():
    torch.manual_seed(195)
    return GPTLanguageModel(
        GPTConfig(7, 6, 8, 2, 1, attention_impl="sdpa", dropout=0.2)
    )


def test_shifted_paired_identical_models_have_zero_deltas_and_lattice_metadata():
    first = model().train()
    second = copy.deepcopy(first).eval()
    rng = torch.get_rng_state().clone()
    result = compare_paired_batches(
        first, second, torch.arange(250) % 7,
        seeds=(1, 2, 3), batch_size=2, block_size=5,
        eval_batches=3, sampling_policy="shifted",
    )
    assert result.sampling_policy == "shifted"
    assert result.per_seed_means == (0.0, 0.0, 0.0)
    assert len(result.lattice_by_seed) == 3
    for overlap, lattice in zip(result.overlap_by_seed, result.lattice_by_seed):
        assert overlap.repeated_window_token_positions == 0
        assert lattice is not None
        assert 0 <= lattice.phase < 6
        assert lattice.selected_tiles == 6
    assert first.training and not second.training
    assert torch.equal(rng, torch.get_rng_state())


def test_default_replacement_keeps_unmodified_behavior():
    first = model()
    second = copy.deepcopy(first)
    result = compare_paired_batches(
        first, second, torch.arange(250) % 7,
        seeds=(2,), batch_size=2, block_size=4,
    )
    assert result.sampling_policy == "replacement"
    assert result.lattice_by_seed == (None,)
    assert result.per_seed_means == (0.0,)


def test_fixed_disjoint_reports_phase_zero():
    first = model()
    report = compare_paired_batches(
        first, first, torch.arange(250) % 7,
        seeds=(1, 2), batch_size=2, block_size=5,
        sampling_policy="disjoint",
    )
    assert all(x.phase == 0 for x in report.lattice_by_seed)


def test_shifted_capacity_fails_explicitly():
    first = model()
    with pytest.raises(ValueError, match="shifted disjoint windows require"):
        compare_paired_batches(
            first, first, torch.arange(19) % 7,
            seeds=(1,), batch_size=2, block_size=4,
            eval_batches=2, sampling_policy="shifted",
        )


def test_saved_checkpoint_comparison_supports_shifted_policy(tmp_path):
    first = model().eval()
    checkpoint = tmp_path / "model.pt"
    save_text_checkpoint(first, CharTokenizer("abcdefg"), checkpoint)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcdefg" * 130, encoding="utf-8")
    report = compare_saved_checkpoints(
        checkpoint, checkpoint, corpus,
        seeds=(4, 5), batch_size=2, block_size=4,
        eval_batches=2, sampling_policy="shifted",
    )
    assert report.sampling_policy == "shifted"
    assert report.paired.per_seed_means == (0.0, 0.0)
    assert all(p is not None for p in report.paired.lattice_by_seed)
    assert all(x.repeated_window_token_positions == 0
               for x in report.paired.overlap_by_seed)
