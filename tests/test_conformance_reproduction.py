"""Recomputation can catch a changed measurement in an intact audit file."""

import json
from unittest.mock import patch

import pytest

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.conformance_reproduction import reproduce_conformance_report
from rebuild_gpt.extended_conformance import run_extended_conformance
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.text_checkpoint import save_text_checkpoint
from rebuild_gpt.tokenizer import CharTokenizer


def _checkpoint(tmp_path):
    corpus = tmp_path / "story.txt"
    corpus.write_text("abba" * 60, encoding="utf-8")
    tokenizer = CharTokenizer(corpus.read_text(encoding="utf-8"))
    model = GPTLanguageModel(GPTConfig(
        tokenizer.vocab_size, 4, 8, 2, 1, attention_impl="fused",
    ))
    saved = tmp_path / "model.pt"
    save_text_checkpoint(model, tokenizer, saved)
    return saved, corpus


def test_real_report_reproduces_all_eight_checks(tmp_path):
    checkpoint, corpus = _checkpoint(tmp_path)
    report = tmp_path / "report.json"
    run_extended_conformance(checkpoint, corpus, seed=19, output_path=report)
    before = report.read_bytes()
    verified = reproduce_conformance_report(report, checkpoint, corpus)
    assert verified["status"] == "reproduced"
    assert verified["reproduced_checks"] == 8
    assert verified["compared_measurements"] > 8
    assert report.read_bytes() == before


def test_changed_measurement_is_caught_by_fresh_comparison(tmp_path):
    checkpoint, corpus = _checkpoint(tmp_path)
    saved = run_extended_conformance(checkpoint, corpus, seed=17)
    report = tmp_path / "report.json"
    saved["checks"]["causal_prefix"]["max_prefix_error"] = 0.75
    report.write_text(json.dumps(saved), encoding="utf-8")
    with patch(
        "rebuild_gpt.conformance_reproduction.run_extended_conformance",
        return_value=run_extended_conformance(checkpoint, corpus, seed=17),
    ):
        with pytest.raises(AssertionError, match="max_prefix_error"):
            reproduce_conformance_report(report, checkpoint, corpus)


def test_invalid_tolerance_rejected_without_loading_artifacts():
    with pytest.raises(ValueError, match="tolerances"):
        reproduce_conformance_report("missing.json", "missing.pt", "missing.txt",
                                     atol=float("nan"))
