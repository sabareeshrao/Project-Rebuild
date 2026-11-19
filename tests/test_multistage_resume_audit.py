"""A passing report requires exact equivalence at all restart points."""

import json
from unittest.mock import patch

import pytest

from rebuild_gpt.multistage_resume_audit import audit_multistage_resume


def _corpus(tmp_path):
    corpus = tmp_path / "data.txt"
    corpus.write_text("abba" * 60, encoding="utf-8")
    return corpus


def test_real_multistage_audit_records_intermediate_checkpoint_evidence(tmp_path):
    output = tmp_path / "runs"
    report = audit_multistage_resume(
        _corpus(tmp_path), output, total_steps=4, stops=(1, 2), seed=9,
    )
    assert report["status"] == "passed"
    assert report["checkpoint_comparison"]["identical"]
    assert [stage["step"] for stage in report["stages"]] == [1, 2, 4]
    assert all(len(stage["sha256"]) == 64 for stage in report["stages"])
    assert len(report["sha256"]["uninterrupted_checkpoint"]) == 64
    assert json.loads((output / "multistage_audit.json").read_text()) == report
    assert not list(output.glob(".multistage_audit.*.tmp"))


def test_failed_rerun_removes_previous_passed_report(tmp_path):
    output = tmp_path / "runs"
    output.mkdir()
    old_report = output / "multistage_audit.json"
    old_report.write_text('{"status": "passed"}', encoding="utf-8")
    with patch(
        "rebuild_gpt.multistage_resume_audit.compare_multistage_resume",
        side_effect=AssertionError("moment drift"),
    ):
        with pytest.raises(AssertionError, match="moment drift"):
            audit_multistage_resume(_corpus(tmp_path), output)
    assert not old_report.exists()
