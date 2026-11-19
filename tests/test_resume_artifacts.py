"""Offline verification catches changed bytes without executing PyTorch."""

from hashlib import sha256
import json

import pytest

from rebuild_gpt.resume_artifacts import verify_multistage_artifacts


def _fixture(tmp_path):
    corpus = tmp_path / "story.txt"
    corpus.write_bytes(b"abba" * 45)
    reference = tmp_path / "uninterrupted.pt"
    reference.write_bytes(b"reference-checkpoint")
    steps = [2, 4, 6]
    stages = []
    for step in steps:
        name = "multistage_final.pt" if step == 6 else f"after_step_{step:04d}.pt"
        file = tmp_path / name
        file.write_bytes(f"stage-{step}".encode())
        stages.append({"step": step, "sha256": sha256(file.read_bytes()).hexdigest()})
    data = {
        "format_version": 1, "status": "passed",
        "settings": {"total_steps": 6, "stops": [2, 4], "seed": 9},
        "sha256": {
            "corpus": sha256(corpus.read_bytes()).hexdigest(),
            "uninterrupted_checkpoint": sha256(reference.read_bytes()).hexdigest(),
        },
        "stages": stages,
        "checkpoint_comparison": {"step": 6, "identical": True},
    }
    audit = tmp_path / "multistage_audit.json"
    audit.write_text(json.dumps(data), encoding="utf-8")
    return corpus, audit


def test_all_recorded_artifacts_can_be_verified(tmp_path):
    corpus, audit = _fixture(tmp_path)
    result = verify_multistage_artifacts(audit, corpus)
    assert result["status"] == "verified"
    assert [x["step"] for x in result["stages"]] == [2, 4, 6]
    assert all(len(x["sha256"]) == 64 for x in result["stages"])


def test_modified_stage_is_detected(tmp_path):
    corpus, audit = _fixture(tmp_path)
    (tmp_path / "after_step_0004.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Stage 4 SHA-256 mismatch"):
        verify_multistage_artifacts(audit, corpus)


def test_modified_corpus_is_detected(tmp_path):
    corpus, audit = _fixture(tmp_path)
    corpus.write_bytes(corpus.read_bytes() + b"!")
    with pytest.raises(ValueError, match="Corpus SHA-256 mismatch"):
        verify_multistage_artifacts(audit, corpus)


def test_missing_full_checkpoint_is_detected(tmp_path):
    corpus, audit = _fixture(tmp_path)
    (tmp_path / "uninterrupted.pt").unlink()
    with pytest.raises(FileNotFoundError, match="Uninterrupted"):
        verify_multistage_artifacts(audit, corpus)
