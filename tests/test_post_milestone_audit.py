"""Verify that the new audit uses actual saved model files and measurements."""

import json
from unittest.mock import patch

import pytest

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.extended_conformance import run_extended_conformance
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.post_milestone_audit import run_post_milestone_audit
from rebuild_gpt.post_milestone_cli import main
from rebuild_gpt.text_checkpoint import save_text_checkpoint
from rebuild_gpt.tokenizer import CharTokenizer


def _bundle(tmp_path):
    corpus = tmp_path / "story.txt"
    corpus.write_text("abba" * 75, encoding="utf-8")
    tokenizer = CharTokenizer(corpus.read_text(encoding="utf-8"))
    model = GPTLanguageModel(GPTConfig(
        tokenizer.vocab_size, 4, 8, 2, 1, attention_impl="fused",
    ))
    saved = tmp_path / "model.pt"
    save_text_checkpoint(model, tokenizer, saved)
    conformance = tmp_path / "extended_conformance.json"
    run_extended_conformance(saved, corpus, output_path=conformance)
    return saved, corpus, conformance


def test_real_saved_model_runs_extra_checks_without_changing_inputs(tmp_path):
    checkpoint, corpus, conformance = _bundle(tmp_path)
    original = (checkpoint.read_bytes(), conformance.read_bytes())
    output = tmp_path / "post_milestone.json"
    report = run_post_milestone_audit(
        checkpoint, corpus, conformance, output_path=output, seed=19,
    )
    assert report["status"] == "passed"
    assert report["numerical_budgets"]["checked_measurements"] == 9
    assert report["gradient_coordinates"]["checked_coordinates"] == 3
    assert report["long_prompt_generation"]["prompt_tokens"] == 6
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert (checkpoint.read_bytes(), conformance.read_bytes()) == original


def test_failed_gradient_check_clears_old_success(tmp_path):
    checkpoint, corpus, conformance = _bundle(tmp_path)
    output = tmp_path / "post_milestone.json"
    output.write_text('{"status": "passed"}', encoding="utf-8")
    with patch(
        "rebuild_gpt.post_milestone_audit.check_gradient_coordinates",
        side_effect=AssertionError("gradient drift"),
    ):
        with pytest.raises(AssertionError, match="gradient drift"):
            run_post_milestone_audit(
                checkpoint, corpus, conformance, output_path=output,
            )
    assert not output.exists()


def test_audit_will_not_replace_its_source_report(tmp_path):
    checkpoint, corpus, conformance = _bundle(tmp_path)
    with pytest.raises(ValueError, match="overwrite"):
        run_post_milestone_audit(
            checkpoint, corpus, conformance, output_path=conformance,
        )
    assert conformance.is_file()


def test_cli_forwards_explicit_saved_file_paths(capsys):
    report = {
        "status": "passed",
        "gradient_coordinates": {"checked_coordinates": 3},
        "numerical_budgets": {"checked_measurements": 9},
        "long_prompt_generation": {"block_size": 4},
    }
    with patch(
        "rebuild_gpt.post_milestone_cli.run_post_milestone_audit",
        return_value=report,
    ) as run:
        assert main([
            "--checkpoint", "m.pt", "--corpus", "c.txt",
            "--conformance", "conformance.json", "--output", "audit.json",
            "--seed", "41", "--new-tokens", "5",
        ]) == 0
    run.assert_called_once_with(
        "m.pt", "c.txt", "conformance.json",
        output_path="audit.json", seed=41, new_tokens=5,
    )
    assert "passed" in capsys.readouterr().out


def test_cli_does_not_claim_success_when_source_file_changed(capsys):
    with patch(
        "rebuild_gpt.post_milestone_cli.run_post_milestone_audit",
        side_effect=ValueError("SHA-256 mismatch"),
    ):
        with pytest.raises(SystemExit) as error:
            main([])
    assert error.value.code == 2
    assert "SHA-256 mismatch" in capsys.readouterr().err
