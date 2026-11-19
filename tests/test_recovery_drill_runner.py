"""A real disposable GPT training drill returns to identical final state."""

from hashlib import sha256

import pytest

from rebuild_gpt.multistage_resume_audit import audit_multistage_resume
from rebuild_gpt.recovery_drill_runner import run_recovery_drill
from rebuild_gpt.resume_artifacts import verify_multistage_artifacts


def _prepared_original(tmp_path):
    corpus = tmp_path / "story.txt"
    corpus.write_text("abba" * 65, encoding="utf-8")
    source = tmp_path / "original"
    audit_multistage_resume(
        corpus, source, total_steps=4, stops=(1, 2), seed=7,
    )
    return corpus, source


def test_copy_fault_replay_and_independent_audit_match_original(tmp_path):
    corpus, source = _prepared_original(tmp_path)
    baseline = {
        p.name: sha256(p.read_bytes()).hexdigest()
        for p in source.iterdir() if p.is_file()
    }
    destination = tmp_path / "drill"
    report = run_recovery_drill(
        source, corpus, destination, damage_step=2,
    )
    assert report["status"] == "passed"
    assert report["fallback_step"] == 1
    assert report["independent_check"]["status"] == "verified"
    assert (destination / "recovery_audit.json").is_file()
    assert (destination / "recovered.pt").is_file()
    assert verify_multistage_artifacts(
        source / "multistage_audit.json", corpus,
    )["status"] == "verified"
    assert {
        p.name: sha256(p.read_bytes()).hexdigest()
        for p in source.iterdir() if p.is_file()
    } == baseline


def test_drill_rejects_no_fallback_before_creating_directory(tmp_path):
    corpus, source = _prepared_original(tmp_path)
    destination = tmp_path / "drill"
    with pytest.raises(ValueError, match="latest intermediate"):
        run_recovery_drill(source, corpus, destination, damage_step=1)
    assert not destination.exists()
