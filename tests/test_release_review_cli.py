"""The milestone CLI separates training, hash-only verification and replay."""

from pathlib import Path
from unittest.mock import patch

import pytest

from rebuild_gpt.release_review_cli import main


def test_train_mode_uses_explicit_learning_settings(capsys):
    result = {
        "status": "passed", "checks_passed": 8, "checkpoint": "out/run/capstone/model.pt",
    }
    with patch(
        "rebuild_gpt.release_review_cli.run_review_capstone",
        return_value=result,
    ) as train:
        assert main([
            "--mode", "train", "--corpus", "text.txt", "--output-dir", "out/run",
            "--steps", "3", "--training-seed", "5",
            "--prompt", "b", "--new-tokens", "2",
            "--conformance-seed", "17", "--batch-size", "3",
        ]) == 0
    args = train.call_args
    assert args.args == ("text.txt", "out/run")
    assert args.kwargs["spec"].steps == 3
    assert args.kwargs["spec"].seed == 5
    assert args.kwargs["spec"].prompt == "b"
    assert args.kwargs["spec"].new_tokens == 2
    assert args.kwargs["conformance_seed"] == 17
    assert args.kwargs["batch_size"] == 3
    assert "passed" in capsys.readouterr().out


def test_default_verify_mode_does_not_train_or_reproduce(capsys):
    report = {
        "status": "verified", "conformance_checks": 8,
        "checkpoint_sha256": "a" * 64,
    }
    with (
        patch("rebuild_gpt.release_review_cli.verify_review_evidence",
              return_value=report) as verify,
        patch("rebuild_gpt.release_review_cli.run_review_capstone") as train,
        patch("rebuild_gpt.release_review_cli.reproduce_review_evidence") as replay,
    ):
        assert main(["--corpus", "story.txt", "--output-dir", "out/run"]) == 0
    verify.assert_called_once_with("story.txt", "out/run")
    train.assert_not_called()
    replay.assert_not_called()
    assert "verified" in capsys.readouterr().out


def test_reproduce_mode_calls_existing_saved_evidence(capsys):
    report = {
        "status": "reproduced", "reproduced_checks": 8,
        "compared_measurements": 62,
    }
    with patch(
        "rebuild_gpt.release_review_cli.reproduce_review_evidence",
        return_value=report,
    ) as reproduce:
        assert main([
            "--mode", "reproduce", "--corpus", "story.txt",
            "--output-dir", "out/review",
        ]) == 0
    reproduce.assert_called_once_with("story.txt", "out/review")
    assert "62" in capsys.readouterr().out


@pytest.mark.parametrize("mode,exception", [
    ("train", FileExistsError("existing output")),
    ("verify", ValueError("SHA-256 mismatch")),
    ("reproduce", AssertionError("gradient mismatch")),
])
def test_failures_never_print_passing_status(mode, exception, capsys):
    name = {
        "train": "run_review_capstone",
        "verify": "verify_review_evidence",
        "reproduce": "reproduce_review_evidence",
    }[mode]
    with patch(f"rebuild_gpt.release_review_cli.{name}", side_effect=exception):
        with pytest.raises(SystemExit) as err:
            main(["--mode", mode])
    assert err.value.code == 2
    assert str(exception) in capsys.readouterr().err


def test_incomplete_train_report_not_marked_as_success(capsys):
    with patch(
        "rebuild_gpt.release_review_cli.run_review_capstone",
        return_value={"status": "passed", "checks_passed": 7},
    ):
        with pytest.raises(SystemExit) as err:
            main(["--mode", "train"])
    assert err.value.code == 2
    assert "all checks" in capsys.readouterr().err
