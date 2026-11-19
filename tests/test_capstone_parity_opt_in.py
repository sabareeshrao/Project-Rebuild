"""Optional real-model parity: never download or execute upstream code in CI."""

import os
from pathlib import Path

import pytest

from rebuild_gpt.capstone_parity_audit import run_capstone_parity_audit
from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.text_checkpoint import save_text_checkpoint
from rebuild_gpt.tokenizer import CharTokenizer


SOURCE = os.environ.get("NANOGPT_MODEL_PY", "")
pytestmark = pytest.mark.skipif(
    not SOURCE or not Path(SOURCE).is_file(),
    reason="Set NANOGPT_MODEL_PY to a trusted original nanoGPT model.py",
)


def test_real_upstream_matches_saved_capstone_weights_and_samples(tmp_path):
    corpus = tmp_path / "text.txt"
    corpus.write_text("abba " * 60, encoding="utf-8")
    tok = CharTokenizer(corpus.read_text(encoding="utf-8"))
    cfg = GPTConfig(
        vocab_size=tok.vocab_size, block_size=4,
        n_embd=8, n_head=2, n_layer=1,
        attention_impl="fused",
    )
    model = GPTLanguageModel(cfg)
    checkpoint = tmp_path / "model.pt"
    save_text_checkpoint(model, tok, checkpoint)

    report = run_capstone_parity_audit(
        checkpoint, corpus, SOURCE, output_path=tmp_path / "parity.json",
        prompt="a", new_tokens=3, top_k=1, seed=3,
    )
    assert report["status"] == "passed"
    assert report["checks"]["forward_logits_and_loss"]["max_logit_error"] < 2e-5
    assert report["checks"]["forward_logits_and_loss"]["loss_error"] < 2e-5
