"""Four GPT semantics checks should work on one real saved tiny model."""

import json
from unittest.mock import patch

import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.model_semantics import run_model_semantics
from rebuild_gpt.model_semantics_cli import main
from rebuild_gpt.text_checkpoint import save_text_checkpoint
from rebuild_gpt.tokenizer import CharTokenizer


def _files(tmp_path):
    corpus = tmp_path / "story.txt"
    corpus.write_text("abba" * 65, encoding="utf-8")
    tokenizer = CharTokenizer(corpus.read_text(encoding="utf-8"))
    model = GPTLanguageModel(GPTConfig(
        tokenizer.vocab_size, 4, 8, 2, 1, attention_impl="fused",
    ))
    checkpoint = tmp_path / "model.pt"
    save_text_checkpoint(model, tokenizer, checkpoint)
    return checkpoint, corpus


def test_real_checkpoint_runs_all_four_checks_and_preserves_inputs(tmp_path):
    checkpoint, corpus = _files(tmp_path)
    before = checkpoint.read_bytes()
    rng = torch.get_rng_state().clone()
    output = tmp_path / "semantics.json"
    result = run_model_semantics(
        checkpoint, corpus, seed=29, output_path=output,
    )
    assert result["status"] == "passed"
    assert set(result["checks"]) == {
        "dropout", "ignored_target_gradients",
        "tied_weight_gradients", "next_token_distribution",
    }
    assert result["checks"]["dropout"]["training_output_difference"] > 0
    assert result["checks"]["ignored_target_gradients"]["ignored_positions"] == 2
    assert result["checks"]["tied_weight_gradients"]["shared_gradient_error"] < 2e-5
    assert sum(result["checks"]["next_token_distribution"]["probabilities"]) == pytest.approx(1)
    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert checkpoint.read_bytes() == before
    assert torch.equal(torch.get_rng_state(), rng)


def test_failed_semantics_check_clears_stale_success(tmp_path):
    checkpoint, corpus = _files(tmp_path)
    output = tmp_path / "semantics.json"
    output.write_text('{"status": "passed"}', encoding="utf-8")
    with patch(
        "rebuild_gpt.model_semantics.check_masked_target_gradients",
        side_effect=AssertionError("bad masked gradient"),
    ):
        with pytest.raises(AssertionError, match="bad masked"):
            run_model_semantics(checkpoint, corpus, output_path=output)
    assert not output.exists()


def test_audit_refuses_overwriting_its_own_checkpoint(tmp_path):
    checkpoint, corpus = _files(tmp_path)
    with pytest.raises(ValueError, match="overwrite"):
        run_model_semantics(
            checkpoint, corpus, output_path=checkpoint,
        )
    assert checkpoint.is_file()


def test_cli_passes_checkpoint_corpus_and_seed(capsys):
    result = {"status": "passed", "checks": {
        "dropout": {}, "ignored_target_gradients": {},
        "tied_weight_gradients": {}, "next_token_distribution": {},
    }}
    with patch(
        "rebuild_gpt.model_semantics_cli.run_model_semantics",
        return_value=result,
    ) as run:
        assert main([
            "--checkpoint", "gpt.pt", "--corpus", "text.txt",
            "--seed", "27", "--output", "evidence.json",
        ]) == 0
    run.assert_called_once_with(
        "gpt.pt", "text.txt", seed=27, output_path="evidence.json",
    )
    assert "passed" in capsys.readouterr().out


def test_cli_fails_closed_when_gradient_comparison_fails(capsys):
    with patch(
        "rebuild_gpt.model_semantics_cli.run_model_semantics",
        side_effect=AssertionError("weight gradient mismatch"),
    ):
        with pytest.raises(SystemExit) as exc:
            main([])
    assert exc.value.code == 2
    assert "gradient mismatch" in capsys.readouterr().err


def test_tied_weight_stage_receives_real_targets_not_masked_labels(tmp_path):
    from rebuild_gpt.tied_gradient_decomposition import (
        check_tied_gradient_decomposition,
    )

    checkpoint, corpus = _files(tmp_path)
    with patch(
        "rebuild_gpt.model_semantics.check_tied_gradient_decomposition",
        wraps=check_tied_gradient_decomposition,
    ) as checked:
        report = run_model_semantics(checkpoint, corpus, seed=13)
    assert report["status"] == "passed"
    checked.assert_called_once()
    forwarded_targets = checked.call_args.args[2]
    assert bool((forwarded_targets >= 0).all())
