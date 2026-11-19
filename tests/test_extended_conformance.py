"""One saved GPT checkpoint must pass all eight independent checks."""

import json
from unittest.mock import patch

import pytest

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.extended_conformance import run_extended_conformance
from rebuild_gpt.extended_conformance_cli import main
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.text_checkpoint import save_text_checkpoint
from rebuild_gpt.tokenizer import CharTokenizer


def _checkpoint(tmp_path, *, attention_impl="fused"):
    text = "abba" * 80
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(text, encoding="utf-8")
    tok = CharTokenizer(text)
    model = GPTLanguageModel(GPTConfig(
        tok.vocab_size, 4, 8, 2, 1, attention_impl=attention_impl,
    ))
    checkpoint = tmp_path / "model.pt"
    save_text_checkpoint(model, tok, checkpoint)
    return checkpoint, corpus


def test_real_saved_checkpoint_passes_eight_checks(tmp_path):
    checkpoint, corpus = _checkpoint(tmp_path)
    path = tmp_path / "audit.json"
    report = run_extended_conformance(
        checkpoint, corpus, seed=17, output_path=path,
    )
    assert report["status"] == "passed"
    assert len(report["checks"]) == 8
    assert report["checks"]["embedding_tying"]["tied"]
    assert report["checks"]["seeded_generation"]["repeatable"]
    assert report["checks"]["checkpoint_roundtrip"]["identical"]
    assert report["checks"]["attention_backends"]["max_gradient_error"] < 2e-5
    assert json.loads(path.read_text(encoding="utf-8")) == report


def test_failure_clears_stale_passed_audit(tmp_path):
    checkpoint, corpus = _checkpoint(tmp_path)
    path = tmp_path / "audit.json"
    path.write_text('{"status": "passed"}', encoding="utf-8")
    with patch(
        "rebuild_gpt.extended_conformance.check_embedding_tying",
        side_effect=AssertionError("tied weight mismatch"),
    ):
        with pytest.raises(AssertionError, match="tied weight mismatch"):
            run_extended_conformance(
                checkpoint, corpus, output_path=path,
            )
    assert not path.exists()


def test_educational_attention_is_explicitly_excluded(tmp_path):
    checkpoint, corpus = _checkpoint(tmp_path, attention_impl="educational")
    with pytest.raises(ValueError, match="fused or sdpa"):
        run_extended_conformance(checkpoint, corpus)


def test_cli_forwards_inputs_and_report_path(capsys):
    fake = {"status": "passed", "checks": {"embedding_tying": {"tied": True}}}
    with patch(
        "rebuild_gpt.extended_conformance_cli.run_extended_conformance",
        return_value=fake,
    ) as run:
        assert main([
            "--checkpoint", "saved/model.pt",
            "--corpus", "data/text.txt",
            "--seed", "13", "--batch-size", "3",
            "--report", "out/checked.json",
        ]) == 0
    run.assert_called_once_with(
        "saved/model.pt", "data/text.txt",
        seed=13, batch_size=3, output_path="out/checked.json",
    )
    assert "passed" in capsys.readouterr().out


def test_cli_reports_failed_gradient(capsys):
    with patch(
        "rebuild_gpt.extended_conformance_cli.run_extended_conformance",
        side_effect=AssertionError("gradient mismatch"),
    ):
        with pytest.raises(SystemExit) as error:
            main([])
    assert error.value.code == 2
    assert "gradient mismatch" in capsys.readouterr().err
