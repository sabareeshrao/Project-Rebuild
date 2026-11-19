"""A resume audit must never preserve an outdated passing result."""

import json
from unittest.mock import patch

import pytest

from rebuild_gpt.resume_audit import audit_resume_training


def _corpus(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("ab" * 110, encoding="utf-8")
    return path


def test_real_audit_writes_checkpoints_and_hashes_after_matching(tmp_path):
    output = tmp_path / "run"
    report = audit_resume_training(_corpus(tmp_path), output, seed=5)
    assert report["status"] == "passed"
    assert report["checkpoint_comparison"]["identical"]
    assert report["checkpoint_comparison"]["step"] == 4
    assert len(report["sha256"]["corpus"]) == 64
    assert len(report["sha256"]["full"]) == 64
    assert len(report["sha256"]["partial"]) == 64
    assert len(report["sha256"]["resumed"]) == 64
    assert json.loads((output / "resume_audit.json").read_text()) == report


def test_failed_rerun_does_not_leave_stale_success_marker(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    report = output / "resume_audit.json"
    report.write_text('{"status": "passed"}', encoding="utf-8")
    with patch(
        "rebuild_gpt.resume_audit.run_resume_equivalence",
        side_effect=AssertionError("optimizer drift"),
    ):
        with pytest.raises(AssertionError, match="optimizer drift"):
            audit_resume_training(_corpus(tmp_path), output)
    assert not report.exists()
