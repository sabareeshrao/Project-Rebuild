"""The saved text tokenizer determines the prompt for upstream sampling."""

from unittest.mock import patch

import pytest
import torch

from rebuild_gpt.capstone_parity import compare_capstone_generation
from rebuild_gpt.config import GPTConfig
from rebuild_gpt.generation_parity import GenerationParityResult
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.text_checkpoint import save_text_checkpoint
from rebuild_gpt.tokenizer import CharTokenizer


def _checkpoint(tmp_path):
    tok = CharTokenizer("ab a")
    model = GPTLanguageModel(GPTConfig(tok.vocab_size, 4, 8, 2, 1, attention_impl="fused"))
    path = tmp_path / "model.pt"
    save_text_checkpoint(model, tok, path)
    return path, tok


def test_generation_uses_saved_prompt_ids_and_sampling_controls(tmp_path):
    checkpoint, tok = _checkpoint(tmp_path)
    with (
        patch("rebuild_gpt.upstream_parity.load_upstream_model", return_value=object()) as load,
        patch(
            "rebuild_gpt.generation_parity.compare_generation_against_upstream",
            return_value=GenerationParityResult(18, 5, 0.7, 2),
        ) as compare,
    ):
        report = compare_capstone_generation(
            checkpoint, "trusted_model.py", prompt="ab",
            new_tokens=5, temperature=0.7, top_k=2, seed=11,
        )
    assert report == {
        "matched_tensors": 18, "prompt_tokens": 2, "new_tokens": 5,
        "temperature": 0.7, "top_k": 2,
    }
    assert load.call_args.args[0] == "trusted_model.py"
    args, kwargs = compare.call_args
    assert torch.equal(args[2], torch.tensor([tok.encode("ab")]))
    assert args[3] == 5
    assert kwargs == {"temperature": 0.7, "top_k": 2, "seed": 11}


def test_unknown_prompt_character_fails_before_importing_upstream(tmp_path):
    checkpoint, _ = _checkpoint(tmp_path)
    with patch("rebuild_gpt.upstream_parity.load_upstream_model") as loader:
        with pytest.raises(ValueError, match="Character"):
            compare_capstone_generation(checkpoint, "trusted_model.py", prompt="z")
    loader.assert_not_called()


def test_empty_prompt_is_rejected(tmp_path):
    checkpoint, _ = _checkpoint(tmp_path)
    with pytest.raises(ValueError, match="non-empty"):
        compare_capstone_generation(checkpoint, "trusted_model.py", prompt="")
