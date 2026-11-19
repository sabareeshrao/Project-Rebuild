"""A saved character vocabulary and held-out tokens define parity inputs."""

import pytest
import torch

from rebuild_gpt.capstone_parity import prepare_validation_parity_batch
from rebuild_gpt.config import GPTConfig
from rebuild_gpt.tokenizer import CharTokenizer


def _corpus(tmp_path):
    path = tmp_path / "story.txt"
    path.write_text("abba" * 45, encoding="utf-8")
    return path, CharTokenizer(path.read_text(encoding="utf-8"))


def test_validation_inputs_are_seeded_and_shifted(tmp_path):
    path, tok = _corpus(tmp_path)
    config = GPTConfig(tok.vocab_size, 4, 8, 2, 1, attention_impl="fused")
    a, y = prepare_validation_parity_batch(path, tok, config, batch_size=3, seed=13)
    b, z = prepare_validation_parity_batch(path, tok, config, batch_size=3, seed=13)
    assert a.shape == y.shape == (3, 4)
    assert torch.equal(a, b) and torch.equal(y, z)
    assert torch.equal(a[:, 1:], y[:, :-1])
    assert torch.equal(y[:, -1], torch.full((3,), -1, dtype=torch.long))


def test_unmasked_targets_keep_all_next_tokens(tmp_path):
    path, tok = _corpus(tmp_path)
    config = GPTConfig(tok.vocab_size, 4, 8, 2, 1)
    _, targets = prepare_validation_parity_batch(path, tok, config, mask_last=False)
    assert torch.all(targets >= 0)


def test_changed_vocabulary_is_rejected(tmp_path):
    path, tok = _corpus(tmp_path)
    config = GPTConfig(tok.vocab_size, 4, 8, 2, 1)
    with pytest.raises(ValueError, match="vocabulary"):
        prepare_validation_parity_batch(path, CharTokenizer("abc"), config)


@pytest.mark.parametrize("changes", [
    {"batch_size": 0}, {"seed": -1}, {"mask_last": 1},
])
def test_invalid_parity_settings_raise(tmp_path, changes):
    path, tok = _corpus(tmp_path)
    config = GPTConfig(tok.vocab_size, 4, 8, 2, 1)
    with pytest.raises(ValueError):
        prepare_validation_parity_batch(path, tok, config, **changes)
