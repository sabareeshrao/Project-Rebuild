"""CLI separates read-only verification from a disposable training drill."""

from pathlib import Path
from unittest.mock import patch

import pytest

from rebuild_gpt.recovery_lab_cli import main


def test_check_mode_does_not_run_recovery_or_training(capsys):
    report = {"status": "verified", "restart_step": 2, "target_step": 6}
    with (
        patch("rebuild_gpt.recovery_lab_cli.verify_recovery_audit",
              return_value=report) as check,
        patch("rebuild_gpt.recovery_lab_cli.run_recovery_drill") as drill,
    ):
        assert main([
            "--mode", "check", "--source-dir", "out/run",
            "--corpus", "story.txt",
        ]) == 0
    check.assert_called_once_with(Path("out/run/recovery_audit.json"), "story.txt")
    drill.assert_not_called()
    assert "verified" in capsys.readouterr().out


def test_drill_mode_passes_disposable_destination_and_damage_step(capsys):
    result = {
        "status": "passed", "damage_step": 4, "fallback_step": 2,
        "independent_check": {"status": "verified"},
    }
    with patch(
        "rebuild_gpt.recovery_lab_cli.run_recovery_drill",
        return_value=result,
    ) as drill:
        assert main([
            "--mode", "drill", "--source-dir", "out/original",
            "--drill-dir", "out/disposable", "--corpus", "story.txt",
            "--damage-step", "4",
        ]) == 0
    drill.assert_called_once_with(
        "out/original", "story.txt", "out/disposable", damage_step=4,
    )
    output = capsys.readouterr().out
    assert "passed" in output and "Fallback restart step: 2" in output


def test_damaged_checkpoint_cannot_produce_success_message(capsys):
    with patch(
        "rebuild_gpt.recovery_lab_cli.verify_recovery_audit",
        side_effect=ValueError("Recovered checkpoint SHA-256 mismatch"),
    ):
        with pytest.raises(SystemExit) as error:
            main([])
    assert error.value.code == 2
    assert "SHA-256 mismatch" in capsys.readouterr().err


def test_existing_drill_directory_does_not_get_overwritten(capsys):
    with patch(
        "rebuild_gpt.recovery_lab_cli.run_recovery_drill",
        side_effect=FileExistsError("drill output already exists"),
    ):
        with pytest.raises(SystemExit) as error:
            main(["--mode", "drill"])
    assert error.value.code == 2
    assert "already exists" in capsys.readouterr().err
