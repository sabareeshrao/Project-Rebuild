"""A passing grouped-trajectory report requires verified source and checks."""

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from rebuild_gpt.grouped_training_audit import audit_grouped_training_trajectory
from rebuild_gpt.grouped_training_audit_cli import main


def _files(tmp_path):
    checkpoint, corpus, source = (
        tmp_path / "model.pt", tmp_path / "text.txt", tmp_path / "model.py",
    )
    checkpoint.write_bytes(b"mock checkpoint")
    corpus.write_text("abba" * 20, encoding="utf-8")
    source.write_text("# reviewed source fixture\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return checkpoint, corpus, source, digest


def _mock_comparison():
    return {
        "steps": 3, "group_layout": {"matched_parameters": 4},
        "history": [
            {"step": 1, "max_gradient_error": 0.0, "max_updated_weight_error": 0.0},
            {"step": 2, "max_gradient_error": 0.0, "max_updated_weight_error": 0.0},
            {"step": 3, "max_gradient_error": 0.0, "max_updated_weight_error": 0.0},
        ],
    }


def test_audit_checks_hash_then_saves_successful_comparison(tmp_path):
    checkpoint, corpus, source, digest = _files(tmp_path)
    dest = tmp_path / "report.json"
    with (
        patch("rebuild_gpt.grouped_training_audit.load_text_checkpoint",
              return_value=(SimpleNamespace(config="mock-config"), object())) as loader,
        patch("rebuild_gpt.grouped_training_audit.prepare_validation_parity_batch",
              return_value=(torch.zeros(2, 4), torch.zeros(2, 4))),
        patch("rebuild_gpt.grouped_training_audit.load_upstream_model",
              return_value=object()) as upstream,
        patch("rebuild_gpt.grouped_training_audit.compare_grouped_adamw_trajectory",
              return_value=_mock_comparison()) as comparison,
    ):
        report = audit_grouped_training_trajectory(
            checkpoint, corpus, source,
            upstream_sha256=digest.upper(), output_path=dest,
        )
    assert report["status"] == "passed"
    assert len(report["comparison"]["history"]) == 3
    assert report["sha256"]["upstream_model_py"] == digest
    assert json.loads(dest.read_text(encoding="utf-8")) == report
    loader.assert_called_once()
    upstream.assert_called_once()
    comparison.assert_called_once()


def test_bad_sha256_rejected_before_loading_or_importing(tmp_path):
    checkpoint, corpus, source, digest = _files(tmp_path)
    with patch("rebuild_gpt.grouped_training_audit.load_upstream_model") as upstream:
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            audit_grouped_training_trajectory(
                checkpoint, corpus, source, upstream_sha256="0" * 64,
            )
    upstream.assert_not_called()


def test_invalid_digest_is_rejected(tmp_path):
    checkpoint, corpus, source, _ = _files(tmp_path)
    with pytest.raises(ValueError, match="64 hexadecimal"):
        audit_grouped_training_trajectory(
            checkpoint, corpus, source, upstream_sha256="short",
        )


def test_failed_trajectory_does_not_write_success_report(tmp_path):
    checkpoint, corpus, source, digest = _files(tmp_path)
    dest = tmp_path / "report.json"
    with (
        patch("rebuild_gpt.grouped_training_audit.load_text_checkpoint",
              return_value=(SimpleNamespace(config="mock-config"), object())),
        patch("rebuild_gpt.grouped_training_audit.prepare_validation_parity_batch",
              return_value=(None, None)),
        patch("rebuild_gpt.grouped_training_audit.load_upstream_model",
              return_value=object()),
        patch("rebuild_gpt.grouped_training_audit.compare_grouped_adamw_trajectory",
              side_effect=AssertionError("step 2 mismatch")),
    ):
        with pytest.raises(AssertionError, match="step 2 mismatch"):
            audit_grouped_training_trajectory(
                checkpoint, corpus, source,
                upstream_sha256=digest, output_path=dest,
            )
    assert not dest.exists()


def test_cli_requires_explicit_source_and_digest(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--upstream-model", "example/model.py"])
    assert error.value.code == 2
    assert "--upstream-sha256" in capsys.readouterr().err


def test_cli_forwards_requested_controls(capsys):
    with patch(
        "rebuild_gpt.grouped_training_audit_cli.audit_grouped_training_trajectory",
        return_value=_mock_comparison() | {"status": "passed", "comparison": _mock_comparison()},
    ) as run:
        assert main([
            "--upstream-model", "trusted/model.py",
            "--upstream-sha256", "a" * 64,
            "--steps", "4",
        ]) == 0
    assert run.call_args.kwargs["steps"] == 4
    assert "passed" in capsys.readouterr().out
