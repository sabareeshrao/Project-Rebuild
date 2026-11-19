"""The conformance suite runs real GPT checks, not just mocked functions."""

import json
from unittest.mock import patch

import pytest

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.model_conformance import run_model_conformance
from rebuild_gpt.model_conformance_cli import main
from rebuild_gpt.text_checkpoint import save_text_checkpoint
from rebuild_gpt.tokenizer import CharTokenizer


def _checkpoint(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abba" * 55, encoding="utf-8")
    tokenizer = CharTokenizer(corpus.read_text(encoding="utf-8"))
    model = GPTLanguageModel(GPTConfig(
        tokenizer.vocab_size, 4, 8, 2, 1, attention_impl="fused",
    ))
    checkpoint = tmp_path / "model.pt"
    save_text_checkpoint(model, tokenizer, checkpoint)
    return checkpoint, corpus


def test_real_saved_model_passes_all_four_checks(tmp_path):
    checkpoint, corpus = _checkpoint(tmp_path)
    output = tmp_path / "audit.json"
    report = run_model_conformance(
        checkpoint, corpus, seed=13, output_path=output,
    )
    assert report["status"] == "passed"
    assert set(report["checks"]) == {
        "causal_prefix", "batch_independence",
        "forward_contract", "finite_difference",
    }
    assert report["checks"]["finite_difference"]["absolute_error"] < 1e-5
    assert report["checks"]["forward_contract"]["ignored_positions"] == 2
    assert len(report["sha256"]["checkpoint"]) == 64
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_failed_conformance_removes_old_success_report(tmp_path):
    checkpoint, corpus = _checkpoint(tmp_path)
    report_file = tmp_path / "audit.json"
    report_file.write_text('{"status":"passed"}', encoding="utf-8")
    with patch(
        "rebuild_gpt.model_conformance.check_causal_prefix",
        side_effect=AssertionError("future leakage"),
    ):
        with pytest.raises(AssertionError, match="future leakage"):
            run_model_conformance(
                checkpoint, corpus, output_path=report_file,
            )
    assert not report_file.exists()


@pytest.mark.parametrize("settings", [
    {"batch_size": 1}, {"batch_size": True}, {"seed": -1},
])
def test_invalid_settings_fail_before_checkpoint_loading(tmp_path, settings):
    checkpoint, corpus = _checkpoint(tmp_path)
    with pytest.raises(ValueError):
        run_model_conformance(checkpoint, corpus, **settings)


def test_cli_forwards_model_and_corpus_paths(capsys):
    fake = {"status": "passed", "checks": {"causal_prefix": {"max_prefix_error": 0.0}}}
    with patch(
        "rebuild_gpt.model_conformance_cli.run_model_conformance",
        return_value=fake,
    ) as run:
        assert main([
            "--checkpoint", "run/model.pt", "--corpus", "story.txt",
            "--batch-size", "3", "--seed", "7", "--report", "out/audit.json",
        ]) == 0
    run.assert_called_once_with(
        "run/model.pt", "story.txt", batch_size=3,
        seed=7, output_path="out/audit.json",
    )
    assert "passed" in capsys.readouterr().out


def test_cli_surfaces_failed_invariant(capsys):
    with patch(
        "rebuild_gpt.model_conformance_cli.run_model_conformance",
        side_effect=AssertionError("causal leak"),
    ):
        with pytest.raises(SystemExit) as exc:
            main([])
    assert exc.value.code == 2
    assert "causal leak" in capsys.readouterr().err
