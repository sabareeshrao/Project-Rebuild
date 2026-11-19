"""The CLI exposes a safe interruption schedule and reports failures."""

from unittest.mock import patch

import pytest

from rebuild_gpt.multistage_resume_cli import main, parse_stops


def test_parse_stops_is_strict():
    assert parse_stops("2, 4") == (2, 4)
    for invalid in ("", "1,", "1,,3", "1;3", "-1,2", "a,b"):
        with pytest.raises(ValueError, match="comma-separated"):
            parse_stops(invalid)


def test_cli_forwards_requested_schedule(capsys):
    report = {
        "status": "passed",
        "stages": [{"step": 2}, {"step": 5}, {"step": 7}],
        "checkpoint_comparison": {"identical": True},
    }
    with patch(
        "rebuild_gpt.multistage_resume_cli.audit_multistage_resume",
        return_value=report,
    ) as audit:
        assert main([
            "--corpus", "story.txt", "--output-dir", "out/run",
            "--total-steps", "7", "--stops", "2,5", "--seed", "17",
        ]) == 0
    audit.assert_called_once_with(
        "story.txt", "out/run", total_steps=7, stops=(2, 5), seed=17,
    )
    output = capsys.readouterr().out
    assert "passed" in output
    assert "out/run/multistage_audit.json" in output


def test_cli_rejects_malformed_schedule(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--stops", "2,"])
    assert exc.value.code == 2
    assert "comma-separated" in capsys.readouterr().err


def test_cli_does_not_claim_success_on_failed_state_comparison(capsys):
    with patch(
        "rebuild_gpt.multistage_resume_cli.audit_multistage_resume",
        side_effect=AssertionError("checkpoint model weights differ"),
    ):
        with pytest.raises(SystemExit) as exc:
            main([])
    assert exc.value.code == 2
    assert "weights differ" in capsys.readouterr().err
