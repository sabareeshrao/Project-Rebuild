"""Audit a real saved GPT without modifying the checkpoint or global RNG."""

import json
from unittest.mock import patch

import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.next_token_loss_audit import run_next_token_loss_audit
from rebuild_gpt.next_token_loss_cli import main
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


def test_real_checkpoint_passes_four_loss_checks_on_unequal_masks(tmp_path):
    checkpoint, corpus = _files(tmp_path)
    original = checkpoint.read_bytes()
    rng = torch.get_rng_state().clone()
    output = tmp_path / "loss.json"
    result = run_next_token_loss_audit(
        checkpoint, corpus, seed=17, output_path=output,
    )
    assert result["status"] == "passed"
    assert len(result["checks"]) == 4
    assert result["checks"]["logit_shift"]["status"] == "invariant"
    assert result["checks"]["masked_microbatch"]["status"] == "matched"
    assert result["checks"]["masked_microbatch"]["microbatch_supervised_counts"] == [1, 3]
    assert result["checks"]["uniform_perplexity"]["supervised_positions"] == 4
    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert checkpoint.read_bytes() == original
    assert torch.equal(torch.get_rng_state(), rng)


def test_failed_loss_check_removes_stale_success(tmp_path):
    checkpoint, corpus = _files(tmp_path)
    audit = tmp_path / "loss.json"
    audit.write_text('{"status":"passed"}', encoding="utf-8")
    with patch(
        "rebuild_gpt.next_token_loss_audit.check_masked_microbatch_equivalence",
        side_effect=AssertionError("gradient mismatch"),
    ):
        with pytest.raises(AssertionError, match="gradient mismatch"):
            run_next_token_loss_audit(checkpoint, corpus, output_path=audit)
    assert not audit.exists()


def test_unsafe_output_path_is_never_overwritten(tmp_path):
    checkpoint, corpus = _files(tmp_path)
    with pytest.raises(ValueError, match="overwrite"):
        run_next_token_loss_audit(checkpoint, corpus, output_path=checkpoint)
    assert checkpoint.is_file()


@pytest.mark.parametrize("options", [{"seed": True}, {"seed": -1}, {"batch_size": 1}])
def test_invalid_settings_fail_before_checkpoint_loading(options):
    with pytest.raises(ValueError):
        run_next_token_loss_audit("missing.pt", "missing.txt", **options)


def test_cli_forwards_explicit_paths_and_seed(capsys):
    fake = {"status": "passed", "checks": {
        "uniform_perplexity": {"supervised_positions": 4, "perplexity": 2.0},
        "masked_microbatch": {"microbatch_supervised_counts": [1, 3]},
    }}
    with patch(
        "rebuild_gpt.next_token_loss_cli.run_next_token_loss_audit",
        return_value=fake,
    ) as run:
        assert main([
            "--checkpoint", "m.pt", "--corpus", "c.txt",
            "--output", "audit.json", "--seed", "7", "--batch-size", "3",
        ]) == 0
    run.assert_called_once_with(
        "m.pt", "c.txt", seed=7, batch_size=3, output_path="audit.json",
    )
    assert "passed" in capsys.readouterr().out


def test_cli_refuses_claiming_success_after_failure(capsys):
    with patch(
        "rebuild_gpt.next_token_loss_cli.run_next_token_loss_audit",
        side_effect=AssertionError("weighted loss mismatch"),
    ):
        with pytest.raises(SystemExit) as exc:
            main([])
    assert exc.value.code == 2
    assert "weighted loss mismatch" in capsys.readouterr().err
