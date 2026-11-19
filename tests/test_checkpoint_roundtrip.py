"""Saving a GPT twice must preserve weights, vocabulary, and predictions."""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from rebuild_gpt.checkpoint_roundtrip import check_checkpoint_roundtrip
from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.text_checkpoint import save_text_checkpoint
from rebuild_gpt.tokenizer import CharTokenizer


def _checkpoint(tmp_path):
    tokenizer = CharTokenizer("abcabc")
    model = GPTLanguageModel(GPTConfig(
        tokenizer.vocab_size, 4, 8, 2, 1, attention_impl="fused",
    ))
    path = tmp_path / "saved.pt"
    save_text_checkpoint(model, tokenizer, path)
    return path


def test_roundtrip_keeps_every_tensor_and_prediction_identical(tmp_path):
    saved = _checkpoint(tmp_path)
    before = saved.read_bytes()
    rng = torch.get_rng_state().clone()
    result = check_checkpoint_roundtrip(
        saved, torch.tensor([[0, 1, 2], [2, 1, 0]]),
    )
    assert result["identical"]
    assert result["state_tensors"] > 0
    assert result["checked_logits"] == 2 * 3 * 3
    assert saved.read_bytes() == before
    assert torch.equal(torch.get_rng_state(), rng)


def test_missing_bundle_fails_without_modifying_rng(tmp_path):
    rng = torch.get_rng_state().clone()
    with pytest.raises(FileNotFoundError):
        check_checkpoint_roundtrip(
            tmp_path / "missing.pt", torch.tensor([[0, 1]]),
        )
    assert torch.equal(torch.get_rng_state(), rng)


def test_untied_bundle_is_rejected(tmp_path):
    tok = CharTokenizer("abc")
    model = GPTLanguageModel(GPTConfig(
        3, 4, 8, 2, 1, tie_embeddings=False, attention_impl="fused",
    ))
    file = tmp_path / "untied.pt"
    save_text_checkpoint(model, tok, file)
    with pytest.raises(ValueError, match="tied"):
        check_checkpoint_roundtrip(file, torch.tensor([[0, 1]]))
