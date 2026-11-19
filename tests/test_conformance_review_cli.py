"""The review CLI must clearly separate file checks and model execution."""

from pathlib import Path
from unittest.mock import patch

import pytest

from rebuild_gpt.conformance_review_cli import main, parse_seeds


def test_seed_list_is_strict():
    assert parse_seeds("11, 29,47") == (11, 29, 47)
    for invalid in ("", "11", "11,11", "-1,2", "1,", "1,,2", "abc,2"):
        with pytest.raises(ValueError):
            parse_seeds(invalid)


def test_verify_mode_only_hashes_files(capsys):
    with (
        patch("rebuild_gpt.conformance_review_cli.verify_conformance_integrity",
              return_value={"status": "verified", "check_count": 8}) as verify,
        patch("rebuild_gpt.conformance_review_cli.reproduce_conformance_report") as rerun,
        patch("rebuild_gpt.conformance_review_cli.run_conformance_seed_sweep") as sweep,
    ):
        assert main([
            "--mode", "verify", "--report", "report.json",
            "--checkpoint", "model.pt", "--corpus", "story.txt",
        ]) == 0
    verify.assert_called_once_with("report.json", "model.pt", "story.txt")
    rerun.assert_not_called()
    sweep.assert_not_called()
    assert "verified" in capsys.readouterr().out


def test_reproduce_mode_forwards_source_files(capsys):
    with patch(
        "rebuild_gpt.conformance_review_cli.reproduce_conformance_report",
        return_value={"status": "reproduced", "reproduced_checks": 8},
    ) as rerun:
        assert main([
            "--mode", "reproduce", "--report", "r.json",
            "--checkpoint", "m.pt", "--corpus", "c.txt",
        ]) == 0
    rerun.assert_called_once_with("r.json", "m.pt", "c.txt")
    assert "reproduced" in capsys.readouterr().out


def test_sweep_mode_forwards_reproducible_seed_schedule(capsys):
    with patch(
        "rebuild_gpt.conformance_review_cli.run_conformance_seed_sweep",
        return_value={"status": "passed", "seeds": [3, 7]},
    ) as run:
        assert main([
            "--mode", "sweep", "--checkpoint", "model.pt", "--corpus", "text.txt",
            "--seed-sweep", "3,7", "--batch-size", "4", "--output", "sweep.json",
        ]) == 0
    run.assert_called_once_with(
        "model.pt", "text.txt", seeds=(3, 7),
        batch_size=4, output_path="sweep.json",
    )
    assert "passed" in capsys.readouterr().out


def test_failed_check_never_prints_a_passing_result(capsys):
    with patch(
        "rebuild_gpt.conformance_review_cli.verify_conformance_integrity",
        side_effect=ValueError("Checkpoint SHA-256 mismatch"),
    ):
        with pytest.raises(SystemExit) as error:
            main([])
    assert error.value.code == 2
    assert "SHA-256 mismatch" in capsys.readouterr().err
