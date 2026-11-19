"""The resume audit CLI must expose controls and report failures clearly."""

from unittest.mock import patch

import pytest

from rebuild_gpt.resume_audit_cli import main


def test_cli_forwards_exact_requested_settings_and_reports_result(capsys):
    fake = {
        "status": "passed",
        "checkpoint_comparison": {"matched_tensors": 42, "moment_parameters": 12},
        "evaluation_records": 5,
    }
    with patch("rebuild_gpt.resume_audit_cli.audit_resume_training", return_value=fake) as audit:
        assert main([
            "--corpus", "sample.txt", "--output-dir", "out/experiment",
            "--first-steps", "2", "--total-steps", "5", "--seed", "19",
        ]) == 0
    audit.assert_called_once_with(
        "sample.txt", "out/experiment", first_steps=2, total_steps=5, seed=19,
    )
    out = capsys.readouterr().out
    assert "passed" in out
    assert "out/experiment/resume_audit.json" in out


def test_cli_turns_mismatched_states_into_failure(capsys):
    with patch(
        "rebuild_gpt.resume_audit_cli.audit_resume_training",
        side_effect=AssertionError("AdamW moment drift"),
    ):
        with pytest.raises(SystemExit) as raised:
            main([])
    assert raised.value.code == 2
    assert "AdamW moment drift" in capsys.readouterr().err


def test_cli_rejects_invalid_step_controls(capsys):
    with patch(
        "rebuild_gpt.resume_audit_cli.audit_resume_training",
        side_effect=ValueError("total_steps must exceed first_steps"),
    ):
        with pytest.raises(SystemExit) as raised:
            main(["--first-steps", "4", "--total-steps", "4"])
    assert raised.value.code == 2
    assert "total_steps" in capsys.readouterr().err
