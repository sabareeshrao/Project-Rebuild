"""Optional actual training-step test: only import user-reviewed upstream code."""

import os
from pathlib import Path

import pytest

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.saved_training_parity import compare_saved_capstone_adamw
from rebuild_gpt.text_checkpoint import save_text_checkpoint
from rebuild_gpt.tokenizer import CharTokenizer

SOURCE = os.getenv("NANOGPT_MODEL_PY", "")
pytestmark = pytest.mark.skipif(
    not SOURCE or not Path(SOURCE).is_file(),
    reason="Set NANOGPT_MODEL_PY to a trusted reviewed original nanoGPT/model.py",
)


def test_real_upstream_matches_one_plain_adamw_update(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abba" * 64, encoding="utf-8")
    tokenizer = CharTokenizer(corpus.read_text(encoding="utf-8"))
    model = GPTLanguageModel(GPTConfig(
        tokenizer.vocab_size, 4, 8, 2, 1, attention_impl="fused",
    ))
    checkpoint = tmp_path / "model.pt"
    save_text_checkpoint(model, tokenizer, checkpoint)
    result = compare_saved_capstone_adamw(
        checkpoint, corpus, SOURCE, seed=7, learning_rate=0.001,
    )
    assert result["matched_gradients"] > 0
    assert result["max_gradient_error"] < 2e-5
    assert result["max_updated_weight_error"] < 2e-5
