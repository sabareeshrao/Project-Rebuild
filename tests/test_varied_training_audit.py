"""The SHA-pinned audit can now use fresh training-only batches."""

from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from rebuild_gpt.grouped_training_audit import audit_grouped_training_trajectory
from rebuild_gpt.grouped_training_audit_cli import main


def _paths(tmp_path):
    model = tmp_path / "model.pt"
    corpus = tmp_path / "corpus.txt"
    upstream = tmp_path / "model.py"
    model.write_bytes(b"checkpoint-fixture")
    corpus.write_text("abc" * 100, encoding="utf-8")
    upstream.write_text("# reviewed example\n", encoding="utf-8")
    return model, corpus, upstream, sha256(upstream.read_bytes()).hexdigest()


def _batches():
    return (
        (torch.tensor([[0, 1, 2]]), torch.tensor([[1, 2, 0]])),
        (torch.tensor([[2, 1, 0]]), torch.tensor([[1, 0, 2]])),
    )


def test_train_sequence_is_forwarded_with_independent_step_pairs(tmp_path):
    model, corpus, upstream, digest = _paths(tmp_path)
    batches = _batches()
    with (
        patch("rebuild_gpt.grouped_training_audit.load_text_checkpoint",
              return_value=(SimpleNamespace(config=SimpleNamespace(block_size=3)), object())),
        patch("rebuild_gpt.grouped_training_audit.training_parity_batches",
              return_value=batches) as sampler,
        patch("rebuild_gpt.grouped_training_audit.prepare_validation_parity_batch") as val,
        patch("rebuild_gpt.grouped_training_audit.load_upstream_model",
              return_value=object()),
        patch("rebuild_gpt.grouped_training_audit.compare_grouped_adamw_trajectory",
              return_value={"steps": 2, "history": []}) as compare,
    ):
        report = audit_grouped_training_trajectory(
            model, corpus, upstream, upstream_sha256=digest,
            steps=2, batch_mode="train_sequence",
        )
    assert report["status"] == "passed"
    assert report["settings"]["batch_mode"] == "train_sequence"
    assert "TRAINING batches" in report["scope"]
    assert sampler.call_args.kwargs == {
        "block_size": 3, "steps": 2, "batch_size": 2, "seed": 29,
    }
    val.assert_not_called()
    assert compare.call_args.kwargs["batches"] is batches
    assert torch.equal(compare.call_args.args[2], batches[0][0])


def test_bad_batch_mode_fails_before_loading_checkpoint(tmp_path):
    model, corpus, upstream, digest = _paths(tmp_path)
    with patch("rebuild_gpt.grouped_training_audit.load_text_checkpoint") as loader:
        with pytest.raises(ValueError, match="batch_mode"):
            audit_grouped_training_trajectory(
                model, corpus, upstream,
                upstream_sha256=digest, batch_mode="unknown",
            )
    loader.assert_not_called()


def test_cli_selects_train_sequence(capsys):
    report = {"status": "passed", "comparison": {"history": [{}, {}]}}
    with patch(
        "rebuild_gpt.grouped_training_audit_cli.audit_grouped_training_trajectory",
        return_value=report,
    ) as audit:
        assert main([
            "--upstream-model", "trusted/model.py",
            "--upstream-sha256", "f" * 64,
            "--steps", "2", "--batch-mode", "train_sequence",
        ]) == 0
    assert audit.call_args.kwargs["batch_mode"] == "train_sequence"
    assert "passed" in capsys.readouterr().out
