"""Offline digest checks plus a fresh full-state comparison of recovery."""

from hashlib import sha256
import json
from unittest.mock import patch

import pytest

from rebuild_gpt.recovery_audit_verification import verify_recovery_audit


def _digest(data):
    return sha256(data).hexdigest()


def _fixture(tmp_path):
    corpus = tmp_path / "story.txt"
    corpus.write_bytes(b"abba" * 20)
    reference = tmp_path / "uninterrupted.pt"
    reference.write_bytes(b"original-weights")
    restart = tmp_path / "after_step_0002.pt"
    restart.write_bytes(b"valid-stage")
    recovered = tmp_path / "recovered.pt"
    recovered.write_bytes(b"restored-weights")
    final = tmp_path / "multistage_final.pt"
    final.write_bytes(b"last-stage")
    stages = [
        {"step": 2, "sha256": _digest(restart.read_bytes())},
        {"step": 4, "sha256": _digest(final.read_bytes())},
    ]
    source = {
        "format_version": 1, "status": "passed",
        "settings": {"stops": [2], "total_steps": 4, "seed": 3},
        "sha256": {
            "corpus": _digest(corpus.read_bytes()),
            "uninterrupted_checkpoint": _digest(reference.read_bytes()),
        },
        "stages": stages,
        "checkpoint_comparison": {"identical": True, "step": 4},
    }
    manifest = tmp_path / "multistage_audit.json"
    manifest.write_text(json.dumps(source), encoding="utf-8")
    comparison = {"identical": True, "step": 4, "matched_tensors": 20}
    audit = {
        "format_version": 1, "status": "passed",
        "source_manifest_sha256": _digest(manifest.read_bytes()),
        "recovered_checkpoint_sha256": _digest(recovered.read_bytes()),
        "recovery": {
            "status": "identical", "restart_step": 2, "target_step": 4,
            "restarted_from": str(restart),
            "recovered_checkpoint": str(recovered),
            "skipped_later_stages": [],
            "checkpoint_comparison": comparison,
        },
    }
    audit_path = tmp_path / "recovery_audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    return audit_path, corpus, comparison


def test_verifier_recomputes_digests_and_exact_state_comparison(tmp_path):
    audit, corpus, expected = _fixture(tmp_path)
    with patch(
        "rebuild_gpt.recovery_audit_verification.compare_resume_checkpoints",
        return_value=expected,
    ) as compare:
        report = verify_recovery_audit(audit, corpus)
    assert report["status"] == "verified"
    assert report["restart_step"] == 2
    compare.assert_called_once_with(
        tmp_path / "uninterrupted.pt", tmp_path / "recovered.pt",
    )


def test_changed_recovered_checkpoint_is_rejected(tmp_path):
    audit, corpus, _ = _fixture(tmp_path)
    (tmp_path / "recovered.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Recovered checkpoint SHA-256"):
        verify_recovery_audit(audit, corpus)


def test_modified_source_manifest_is_rejected(tmp_path):
    audit, corpus, _ = _fixture(tmp_path)
    (tmp_path / "multistage_audit.json").write_text("not original", encoding="utf-8")
    with pytest.raises(ValueError, match="Source manifest SHA-256"):
        verify_recovery_audit(audit, corpus)


def test_false_saved_comparison_is_caught_independently(tmp_path):
    audit, corpus, _ = _fixture(tmp_path)
    with patch(
        "rebuild_gpt.recovery_audit_verification.compare_resume_checkpoints",
        return_value={"identical": True, "step": 4, "matched_tensors": 19},
    ):
        with pytest.raises(AssertionError, match="differs from saved audit"):
            verify_recovery_audit(audit, corpus)


def test_wrong_recovery_step_is_rejected_before_loading_torch(tmp_path):
    audit, corpus, _ = _fixture(tmp_path)
    data = json.loads(audit.read_text())
    data["recovery"]["restart_step"] = 3
    audit.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="recorded interruption"):
        verify_recovery_audit(audit, corpus)
