"""Recovery readiness never interprets damaged checkpoint contents."""

from hashlib import sha256
import json

import pytest

from rebuild_gpt.resume_health import inspect_recovery_health


def _fixture(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(b"abba" * 40)
    full = tmp_path / "uninterrupted.pt"
    full.write_bytes(b"reference")
    stages = []
    for step in (2, 4, 6):
        name = "multistage_final.pt" if step == 6 else f"after_step_{step:04d}.pt"
        checkpoint = tmp_path / name
        checkpoint.write_bytes(f"step-{step}".encode())
        stages.append({"step": step, "sha256": sha256(checkpoint.read_bytes()).hexdigest()})
    report = {
        "format_version": 1, "status": "passed",
        "settings": {"total_steps": 6, "stops": [2, 4], "seed": 5},
        "sha256": {
            "corpus": sha256(corpus.read_bytes()).hexdigest(),
            "uninterrupted_checkpoint": sha256(full.read_bytes()).hexdigest(),
        },
        "stages": stages, "checkpoint_comparison": {"step": 6, "identical": True},
    }
    manifest = tmp_path / "multistage_audit.json"
    manifest.write_text(json.dumps(report), encoding="utf-8")
    return corpus, manifest


def test_ready_when_all_intermediate_stages_match(tmp_path):
    corpus, manifest = _fixture(tmp_path)
    report = inspect_recovery_health(manifest, corpus)
    assert report["status"] == "ready"
    assert report["recommended_restart_step"] == 4
    assert report["verified_intermediates"] == 2


def test_degraded_stage_recommends_earlier_fallback(tmp_path):
    corpus, manifest = _fixture(tmp_path)
    (tmp_path / "after_step_0004.pt").write_bytes(b"modified")
    report = inspect_recovery_health(manifest, corpus)
    assert report["status"] == "degraded"
    assert report["recommended_restart_step"] == 2
    assert report["stages"][1]["status"] == "sha256_mismatch"


def test_no_surviving_intermediate_is_unrecoverable(tmp_path):
    corpus, manifest = _fixture(tmp_path)
    for step in (2, 4):
        (tmp_path / f"after_step_{step:04d}.pt").unlink()
    report = inspect_recovery_health(manifest, corpus)
    assert report["status"] == "unrecoverable"
    assert report["recommended_restart_step"] is None


@pytest.mark.parametrize("filename", ["corpus.txt", "uninterrupted.pt"])
def test_changed_corpus_or_reference_is_hard_error(tmp_path, filename):
    corpus, manifest = _fixture(tmp_path)
    (tmp_path / filename).write_bytes(b"unexpected")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        inspect_recovery_health(manifest, corpus)
