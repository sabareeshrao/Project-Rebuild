"""Optional genuine upstream optimizer comparison; CI skips unless opted in."""

import os
from pathlib import Path

import pytest

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.grouped_training_audit import audit_grouped_training_trajectory, sha256_file
from rebuild_gpt.text_checkpoint import save_text_checkpoint
from rebuild_gpt.tokenizer import CharTokenizer


SOURCE = os.environ.get("NANOGPT_MODEL_PY", "")
pytestmark = pytest.mark.skipif(
    not SOURCE or not Path(SOURCE).is_file(),
    reason="Set NANOGPT_MODEL_PY to a personally reviewed original nanoGPT/model.py",
)


def test_original_grouped_optimizer_matches_three_steps(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abba" * 55, encoding="utf-8")
    tokenizer = CharTokenizer(corpus.read_text(encoding="utf-8"))
    model = GPTLanguageModel(GPTConfig(
        tokenizer.vocab_size, 4, 8, 2, 1, attention_impl="fused",
    ))
    checkpoint = tmp_path / "model.pt"
    save_text_checkpoint(model, tokenizer, checkpoint)
    report = audit_grouped_training_trajectory(
        checkpoint, corpus, SOURCE,
        upstream_sha256=sha256_file(SOURCE), steps=3,
    )
    assert report["status"] == "passed"
    assert len(report["comparison"]["history"]) == 3
    assert all(
        row["max_updated_weight_error"] < 2e-5
        for row in report["comparison"]["history"]
    )
