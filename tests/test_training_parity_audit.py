"""No success artifact may be written when an optimizer check fails."""

from hashlib import sha256
import json
from unittest.mock import patch

import pytest

from rebuild_gpt.training_parity_audit import run_training_step_audit
from rebuild_gpt.training_parity_audit_cli import main


def _inputs(tmp_path):
    paths = [tmp_path / "model.pt", tmp_path / "corpus.txt", tmp_path / "model.py"]
    for path, contents in zip(paths, (b"weights", b"ab ab", b"# trusted fixture\n")):
        path.write_bytes(contents)
    return paths


def _comparison():
    return {
        "batch_size": 2, "context_length": 4, "seed": 29,
        "learning_rate": 0.001, "weight_decay": 0.01,
        "matched_gradients": 12, "max_gradient_error": 0.0,
        "matched_parameter_tensors": 13,
        "max_updated_weight_error": 0.0,
        "largest_local_parameter_change": 0.001,
    }


def test_pinned_report_contains_matched_step_and_digests(tmp_path):
    checkpoint, corpus, source = _inputs(tmp_path)
    pin = sha256(source.read_bytes()).hexdigest()
    report_file = tmp_path / "report.json"
    with patch(
        "rebuild_gpt.training_parity_audit.compare_saved_capstone_adamw",
        return_value=_comparison(),
    ) as compare:
        report = run_training_step_audit(
            checkpoint, corpus, source,
            expected_upstream_sha256=pin.upper(), output_path=report_file,
        )
    assert report["status"] == "passed"
    assert report["sha256"]["upstream_model_py"] == pin
    assert report["comparison"]["max_gradient_error"] == 0.0
    assert report == json.loads(report_file.read_text(encoding="utf-8"))
    compare.assert_called_once()


def test_wrong_upstream_hash_rejects_before_reference_import(tmp_path):
    checkpoint, corpus, source = _inputs(tmp_path)
    with patch("rebuild_gpt.training_parity_audit.compare_saved_capstone_adamw") as compare:
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            run_training_step_audit(
                checkpoint, corpus, source, expected_upstream_sha256="0" * 64,
            )
    compare.assert_not_called()


def test_no_report_written_on_comparison_error(tmp_path):
    checkpoint, corpus, source = _inputs(tmp_path)
    report = tmp_path / "audit.json"
    with patch(
        "rebuild_gpt.training_parity_audit.compare_saved_capstone_adamw",
        side_effect=AssertionError("parameter mismatch"),
    ):
        with pytest.raises(AssertionError, match="parameter mismatch"):
            run_training_step_audit(checkpoint, corpus, source, output_path=report)
    assert not report.exists()


def test_nonfinite_result_does_not_create_success_report(tmp_path):
    checkpoint, corpus, source = _inputs(tmp_path)
    report = tmp_path / "audit.json"
    corrupted = _comparison()
    corrupted["max_gradient_error"] = float("nan")
    with patch(
        "rebuild_gpt.training_parity_audit.compare_saved_capstone_adamw",
        return_value=corrupted,
    ):
        with pytest.raises(ValueError, match="Non-finite"):
            run_training_step_audit(checkpoint, corpus, source, output_path=report)
    assert not report.exists()


def test_cli_forwards_reference_controls_and_report_path(capsys):
    with patch(
        "rebuild_gpt.training_parity_audit_cli.run_training_step_audit",
        return_value={"status": "passed", "comparison": {"matched_gradients": 10}},
    ) as audit:
        assert main([
            "--upstream-model", "trusted/model.py",
            "--upstream-sha256", "f" * 64,
            "--checkpoint", "out/model.pt", "--corpus", "data/file.txt",
            "--report", "out/step.json", "--seed", "13",
            "--batch-size", "3", "--learning-rate", "0.005",
            "--weight-decay", "0.0",
        ]) == 0
    assert audit.call_args.args == ("out/model.pt", "data/file.txt", "trusted/model.py")
    assert audit.call_args.kwargs == {
        "expected_upstream_sha256": "f" * 64,
        "output_path": "out/step.json",
        "batch_size": 3,
        "seed": 13,
        "learning_rate": 0.005,
        "weight_decay": 0.0,
    }
    assert "passed" in capsys.readouterr().out


def test_cli_requires_explicit_upstream_reference():
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 2
