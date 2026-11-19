"""Sampling distinct training batches must be deterministic and leak-free."""

import pytest
import torch

from rebuild_gpt.data import load_corpus, sample_batch, split_train_val
from rebuild_gpt.tokenizer import CharTokenizer
from rebuild_gpt.training_parity_batches import training_parity_batches


def _example(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("abcdefghij" * 40, encoding="utf-8")
    return path, CharTokenizer(path.read_text(encoding="utf-8"))


def test_batch_sequence_matches_independent_train_only_sampling(tmp_path):
    path, tokenizer = _example(tmp_path)
    pairs = training_parity_batches(
        path, tokenizer, steps=4, batch_size=3, block_size=5, seed=17,
    )
    tokens, _ = load_corpus(path)
    train, _ = split_train_val(tokens, val_fraction=0.2)
    generator = torch.Generator().manual_seed(17)
    expected = [
        sample_batch(train, batch_size=3, block_size=5, generator=generator)
        for _ in range(4)
    ]
    assert len(pairs) == 4
    for (actual_x, actual_y), (expected_x, expected_y) in zip(pairs, expected):
        assert actual_x.shape == actual_y.shape == (3, 5)
        assert actual_x.dtype == actual_y.dtype == torch.long
        assert torch.equal(actual_x, expected_x)
        assert torch.equal(actual_y, expected_y)


def test_seeded_plan_does_not_modify_global_rng_state(tmp_path):
    path, tokenizer = _example(tmp_path)
    torch.manual_seed(123)
    previous = torch.get_rng_state().clone()
    a = training_parity_batches(path, tokenizer, seed=3)
    b = training_parity_batches(path, tokenizer, seed=3)
    assert torch.equal(torch.get_rng_state(), previous)
    assert all(torch.equal(x[0], y[0]) and torch.equal(x[1], y[1])
               for x, y in zip(a, b))


def test_mismatched_saved_vocabulary_is_rejected(tmp_path):
    path, _ = _example(tmp_path)
    with pytest.raises(ValueError, match="vocabulary"):
        training_parity_batches(path, CharTokenizer("abc"))


@pytest.mark.parametrize("settings", [
    {"steps": 0}, {"steps": True}, {"batch_size": 0},
    {"block_size": -2}, {"seed": -1},
])
def test_invalid_settings_rejected(tmp_path, settings):
    path, tok = _example(tmp_path)
    with pytest.raises(ValueError):
        training_parity_batches(path, tok, **settings)
