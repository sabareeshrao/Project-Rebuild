"""The user-facing CLI must forward the exact parity controls."""

from unittest.mock import patch

import pytest

from rebuild_gpt.capstone_parity_cli import main


def _fake_report():
    return {
        "status": "passed",
        "checks": {"forward_logits_and_loss": {"matched_tensors": 12}},
        "sha256": {"upstream_model_py": "a" * 64},
    }


def test_parity_cli_forwards_pinned_inputs_and_writes_success_summary(capsys):
    with patch(
        "rebuild_gpt.capstone_parity_cli.run_capstone_parity_audit",
        return_value=_fake_report(),
    ) as audit:
        assert main([
            "--upstream-model", "reviewed/model.py",
            "--upstream-sha256", "a" * 64,
            "--checkpoint", "run/model.pt",
            "--corpus", "data/my_story.txt",
            "--report", "out/audit.json",
            "--batch-size", "3",
            "--seed", "17",
            "--prompt", "ab",
            "--new-tokens", "4",
            "--temperature", "0.7",
            "--top-k", "2",
        ]) == 0
    assert audit.call_args.args == (
        "run/model.pt", "data/my_story.txt", "reviewed/model.py",
    )
    assert audit.call_args.kwargs == {
        "output_path": "out/audit.json",
        "expected_upstream_sha256": "a" * 64,
        "seed": 17,
        "batch_size": 3,
        "prompt": "ab",
        "new_tokens": 4,
        "temperature": 0.7,
        "top_k": 2,
    }
    output = capsys.readouterr().out
    assert "passed" in output and "12" in output and "out/audit.json" in output


def test_parity_cli_reports_failed_comparison_as_error(capsys):
    with patch(
        "rebuild_gpt.capstone_parity_cli.run_capstone_parity_audit",
        side_effect=AssertionError("mismatched logits"),
    ):
        with pytest.raises(SystemExit) as error:
            main(["--upstream-model", "trusted/model.py"])
    assert error.value.code == 2
    assert "mismatched logits" in capsys.readouterr().err


def test_parity_cli_requires_explicit_reference(capsys):
    with pytest.raises(SystemExit) as error:
        main([])
    assert error.value.code == 2
    assert "--upstream-model" in capsys.readouterr().err
