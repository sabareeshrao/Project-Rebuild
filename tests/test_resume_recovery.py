"""Choose an earlier known-good checkpoint when a later stage is damaged."""

from hashlib import sha256
import json

import pytest

from rebuild_gpt.resume_recovery import select_verified_restart


def _run(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(b"abba" * 20)
    reference = tmp_path / "uninterrupted.pt"
    reference.write_bytes(b"baseline")
    stages = []
    for step in (2, 4, 6):
        checkpoint = tmp_path / (
            f"after_step_{step:04d}.pt" if step < 6 else "multistage_final.pt"
        )
        checkpoint.write_bytes(str(step).encode())
        stages.append({
            "step": step, "sha256": sha256(checkpoint.read_bytes()).hexdigest(),
        })
    report = {
        "format_version": 1, "status": "passed",
        "settings": {"total_steps": 6, "stops": [2, 4], "seed": 5},
        "sha256": {
            "corpus": sha256(corpus.read_bytes()).hexdigest(),
            "uninterrupted_checkpoint": sha256(reference.read_bytes()).hexdigest(),
        },
        "stages": stages,
        "checkpoint_comparison": {"step": 6, "identical": True},
    }
    manifest = tmp_path / "multistage_audit.json"
    manifest.write_text(json.dumps(report), encoding="utf-8")
    return corpus, manifest


def test_latest_valid_checkpoint_is_selected(tmp_path):
    corpus, manifest = _run(tmp_path)
    selected = select_verified_restart(manifest, corpus)
    assert selected["step"] == 4
    assert selected["target_step"] == 6
    assert selected["checkpoint"].endswith("after_step_0004.pt")
    assert selected["skipped_later_stages"] == []


def test_damaged_checkpoint_falls_back_to_earlier_verified_one(tmp_path):
    corpus, manifest = _run(tmp_path)
    (tmp_path / "after_step_0004.pt").write_bytes(b"corrupted")
    selected = select_verified_restart(manifest, corpus)
    assert selected["step"] == 2
    assert selected["skipped_later_stages"] == [
        {"step": 4, "reason": "sha256_mismatch"},
    ]


def test_missing_latest_checkpoint_is_skipped(tmp_path):
    corpus, manifest = _run(tmp_path)
    (tmp_path / "after_step_0004.pt").unlink()
    selected = select_verified_restart(manifest, corpus)
    assert selected["step"] == 2
    assert selected["skipped_later_stages"][0]["reason"] == "missing"


def test_no_good_recovery_point_fails_closed(tmp_path):
    corpus, manifest = _run(tmp_path)
    for step in (2, 4):
        (tmp_path / f"after_step_{step:04d}.pt").unlink()
    with pytest.raises(FileNotFoundError, match="No hash-verified"):
        select_verified_restart(manifest, corpus)


def test_corpus_change_is_never_ignored(tmp_path):
    corpus, manifest = _run(tmp_path)
    corpus.write_bytes(b"different corpus")
    with pytest.raises(ValueError, match="Corpus SHA-256 mismatch"):
        select_verified_restart(manifest, corpus)


@pytest.mark.parametrize("before", [0, True, 7, -1])
def test_invalid_target_rejected(tmp_path, before):
    corpus, manifest = _run(tmp_path)
    with pytest.raises(ValueError, match="before_step"):
        select_verified_restart(manifest, corpus, before_step=before)
