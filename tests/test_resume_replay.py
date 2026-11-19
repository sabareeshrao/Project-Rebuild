"""Replay from an intact older checkpoint after a later one is damaged."""

from pathlib import Path

import pytest

from rebuild_gpt.multistage_resume_audit import audit_multistage_resume
from rebuild_gpt.resume_replay import replay_verified_restart


def test_corrupted_late_stage_recovers_identical_cpu_training(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abba" * 70, encoding="utf-8")
    directory = tmp_path / "run"
    audit_multistage_resume(
        corpus, directory, total_steps=4, stops=(1, 2), seed=3,
    )
    (directory / "after_step_0002.pt").write_bytes(b"corrupted checkpoint")
    recovered = replay_verified_restart(
        directory / "multistage_audit.json", corpus,
    )
    assert recovered["status"] == "identical"
    assert recovered["restart_step"] == 1
    assert recovered["target_step"] == 4
    assert recovered["checkpoint_comparison"]["identical"]
    assert recovered["skipped_later_stages"] == [
        {"step": 2, "reason": "sha256_mismatch"}
    ]
    assert Path(recovered["recovered_checkpoint"]).is_file()


def test_recovery_refuses_changed_reference_checkpoint(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abba" * 55, encoding="utf-8")
    directory = tmp_path / "run"
    audit_multistage_resume(
        corpus, directory, total_steps=3, stops=(1, 2), seed=4,
    )
    (directory / "uninterrupted.pt").write_bytes(b"fake reference")
    with pytest.raises(ValueError, match="Uninterrupted checkpoint SHA-256"):
        replay_verified_restart(directory / "multistage_audit.json", corpus)
    assert not (directory / "recovered.pt").exists()
