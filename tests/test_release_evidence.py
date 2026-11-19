"""The final milestone requires one connected, internally consistent audit."""

from hashlib import sha256
import json
from unittest.mock import patch

import pytest

from rebuild_gpt.release_evidence import verify_release_evidence


def _bundle(tmp_path):
    folder = tmp_path / "review"
    folder.mkdir()
    source = folder / "extended_conformance.json"
    source.write_bytes(b"saved-eight-check-report")
    digest = sha256(source.read_bytes()).hexdigest()
    reproduction = {
        "format_version": 1,
        "status": "reproduced",
        "source_conformance_sha256": digest,
        "checkpoint_sha256": "a" * 64,
        "corpus_sha256": "b" * 64,
        "training_steps": 2,
        "reproduced_checks": 8,
        "compared_measurements": 41,
    }
    (folder / "reproduction.json").write_text(json.dumps(reproduction))
    training = {
        "status": "verified",
        "conformance_checks": 8,
        "checkpoint_sha256": "a" * 64,
        "corpus_sha256": "b" * 64,
        "training_steps": 2,
    }
    return folder, training


def test_three_stage_evidence_chain_verifies(tmp_path):
    folder, training = _bundle(tmp_path)
    with patch(
        "rebuild_gpt.release_evidence.verify_review_evidence",
        return_value=training,
    ) as verify:
        result = verify_release_evidence("text.txt", folder)
    verify.assert_called_once_with("text.txt", folder)
    assert result["status"] == "verified"
    assert result["checked_invariants"] == 8
    assert len(result["reproduction_report_sha256"]) == 64


@pytest.mark.parametrize("change,reason", [
    (lambda x: x.update(status="failed"), "successful reproduction"),
    (lambda x: x.update(source_conformance_sha256="0" * 64), "conformance report bytes"),
    (lambda x: x.update(checkpoint_sha256="c" * 64), "checkpoint_sha256"),
    (lambda x: x.update(corpus_sha256="c" * 64), "corpus_sha256"),
    (lambda x: x.update(training_steps=3), "training step"),
    (lambda x: x.update(reproduced_checks=7), "eight invariants"),
    (lambda x: x.update(compared_measurements=0), "enough measurements"),
])
def test_inconsistent_report_cannot_pass(tmp_path, change, reason):
    folder, training = _bundle(tmp_path)
    report_path = folder / "reproduction.json"
    report = json.loads(report_path.read_text())
    change(report)
    report_path.write_text(json.dumps(report))
    with patch(
        "rebuild_gpt.release_evidence.verify_review_evidence",
        return_value=training,
    ):
        with pytest.raises(ValueError, match=reason):
            verify_release_evidence("text.txt", folder)


def test_missing_reproduction_is_not_a_success(tmp_path):
    folder, training = _bundle(tmp_path)
    (folder / "reproduction.json").unlink()
    with patch(
        "rebuild_gpt.release_evidence.verify_review_evidence",
        return_value=training,
    ):
        with pytest.raises(FileNotFoundError):
            verify_release_evidence("text.txt", folder)
