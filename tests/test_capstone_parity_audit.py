"""The audit must never mark a missing/failed comparison as 'passed'."""

import hashlib
import json
from unittest.mock import patch

import pytest

from rebuild_gpt.capstone_parity_audit import run_capstone_parity_audit


def _inputs(tmp_path):
    checkpoint = tmp_path / "model.pt"
    corpus = tmp_path / "corpus.txt"
    upstream = tmp_path / "model.py"
    checkpoint.write_bytes(b"fake-checkpoint-for-mocked-checks")
    corpus.write_text("abc abc", encoding="utf-8")
    upstream.write_text("# reviewed reference placeholder\n", encoding="utf-8")
    return checkpoint, corpus, upstream


def test_audit_records_hashes_only_after_both_checks_succeed(tmp_path):
    checkpoint, corpus, upstream = _inputs(tmp_path)
    report_path = tmp_path / "audit" / "report.json"
    digest = hashlib.sha256(upstream.read_bytes()).hexdigest()
    with (
        patch("rebuild_gpt.capstone_parity_audit.compare_capstone_forward",
              return_value={"max_logit_error": 0.00001}) as forward,
        patch("rebuild_gpt.capstone_parity_audit.compare_capstone_generation",
              return_value={"new_tokens": 4}) as generation,
    ):
        report = run_capstone_parity_audit(
            checkpoint, corpus, upstream,
            output_path=report_path, expected_upstream_sha256=digest.upper(),
            new_tokens=4,
        )
    assert report["status"] == "passed"
    assert report["sha256"]["upstream_model_py"] == digest
    assert report["checks"]["generated_token_ids"] == {"new_tokens": 4}
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    forward.assert_called_once()
    generation.assert_called_once()


def test_digest_mismatch_rejects_before_executing_upstream(tmp_path):
    checkpoint, corpus, upstream = _inputs(tmp_path)
    with patch("rebuild_gpt.capstone_parity_audit.compare_capstone_forward") as comparison:
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            run_capstone_parity_audit(
                checkpoint, corpus, upstream,
                expected_upstream_sha256="0" * 64,
            )
    comparison.assert_not_called()


def test_bad_digest_format_rejected(tmp_path):
    checkpoint, corpus, upstream = _inputs(tmp_path)
    with pytest.raises(ValueError, match="64 hexadecimal"):
        run_capstone_parity_audit(
            checkpoint, corpus, upstream, expected_upstream_sha256="123",
        )


def test_forward_failure_never_writes_passed_report(tmp_path):
    checkpoint, corpus, upstream = _inputs(tmp_path)
    dest = tmp_path / "report.json"
    with patch(
        "rebuild_gpt.capstone_parity_audit.compare_capstone_forward",
        side_effect=AssertionError("logits differ"),
    ):
        with pytest.raises(AssertionError, match="logits differ"):
            run_capstone_parity_audit(checkpoint, corpus, upstream, output_path=dest)
    assert not dest.exists()


def test_generation_failure_never_writes_passed_report(tmp_path):
    checkpoint, corpus, upstream = _inputs(tmp_path)
    dest = tmp_path / "report.json"
    with (
        patch("rebuild_gpt.capstone_parity_audit.compare_capstone_forward",
              return_value={}),
        patch("rebuild_gpt.capstone_parity_audit.compare_capstone_generation",
              side_effect=AssertionError("tokens differ")),
    ):
        with pytest.raises(AssertionError, match="tokens differ"):
            run_capstone_parity_audit(checkpoint, corpus, upstream, output_path=dest)
    assert not dest.exists()
