"""Saved-model comparisons reject incompatible text vocabularies."""

import copy

import pytest
import torch

from rebuild_gpt.checkpoint_comparison import compare_saved_checkpoints
from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.text_checkpoint import save_text_checkpoint
from rebuild_gpt.tokenizer import CharTokenizer


def saved(tmp_path, *, alphabet="abc", name="model.pt", update=0.0):
    vocab = CharTokenizer(alphabet)
    torch.manual_seed(185)
    model = GPTLanguageModel(
        GPTConfig(vocab.vocab_size, 5, 8, 2, 1, dropout=0.2, attention_impl="sdpa")
    ).eval()
    with torch.no_grad():
        model.head.linear.weight.add_(update)
    path = tmp_path / name
    save_text_checkpoint(model, vocab, path)
    return path


def corpus(tmp_path, text="abc" * 160):
    path = tmp_path / "corpus.txt"
    path.write_text(text, encoding="utf-8")
    return path


def test_equal_checkpoints_produce_zero_paired_deltas_and_keep_rng(tmp_path):
    path = saved(tmp_path)
    text = corpus(tmp_path)
    rng = torch.get_rng_state().clone()
    result = compare_saved_checkpoints(
        path, path, text, seeds=(3, 4), batch_size=2,
        block_size=4, eval_batches=2,
    )
    assert result.paired.per_seed_means == (0.0, 0.0)
    assert result.directions.batch_directions.tied == 4
    assert result.validation_token_count == int(len("abc" * 160) * 0.2)
    assert result.vocabulary_size == 3
    assert torch.equal(rng, torch.get_rng_state())


def test_changed_checkpoint_reports_paired_losses_and_config(tmp_path):
    original = saved(tmp_path, name="a.pt")
    updated = saved(tmp_path, name="b.pt", update=0.2)
    result = compare_saved_checkpoints(
        original, updated, corpus(tmp_path),
        seeds=(1, 2, 3), batch_size=3, block_size=4, eval_batches=2,
    )
    assert result.baseline_config == result.candidate_config
    assert result.paired.seeds == (1, 2, 3)
    assert len(result.paired.differences) == 3
    assert all(len(row) == 2 for row in result.paired.differences)
    assert result.directions.batch_directions.total == 6


def test_equal_size_but_different_character_vocabulary_rejected(tmp_path):
    original = saved(tmp_path, alphabet="abc", name="a.pt")
    other = saved(tmp_path, alphabet="abd", name="b.pt")
    with pytest.raises(ValueError, match="vocabularies"):
        compare_saved_checkpoints(
            original, other, corpus(tmp_path),
            seeds=(1,), batch_size=2, block_size=3,
        )


def test_unseen_corpus_character_rejected(tmp_path):
    original = saved(tmp_path)
    with pytest.raises(ValueError, match="not in vocabulary"):
        compare_saved_checkpoints(
            original, original, corpus(tmp_path, "abcd" * 50),
            seeds=(1,), batch_size=2, block_size=3,
        )


def test_long_context_rejected(tmp_path):
    original = saved(tmp_path)
    with pytest.raises(ValueError, match="block_size"):
        compare_saved_checkpoints(
            original, original, corpus(tmp_path),
            seeds=(1,), batch_size=2, block_size=6,
        )


def test_missing_file_is_not_treated_as_inference_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError):
        compare_saved_checkpoints(
            tmp_path / "missing.pt", tmp_path / "missing.pt",
            corpus(tmp_path), seeds=(1,), batch_size=2, block_size=3,
        )
