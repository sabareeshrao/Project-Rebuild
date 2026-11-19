"""Corrupt a DISPOSABLE copy and protect the original checkpoints."""

from hashlib import sha256
import json

import pytest

from rebuild_gpt.recovery_drill import prepare_recovery_drill
from rebuild_gpt.resume_artifacts import verify_multistage_artifacts


def _fixture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(b"abba" * 70)
    def write(name, bytes_):
        file = source / name
        file.write_bytes(bytes_)
        return sha256(bytes_).hexdigest()
    ref = write("uninterrupted.pt", b"full")
    stages = []
    for step in (2, 4, 6):
        name = "multistage_final.pt" if step == 6 else f"after_step_{step:04d}.pt"
        digest = write(name, f"step={step}".encode())
        stages.append({"step": step, "sha256": digest})
    report = {
        "format_version": 1, "status": "passed",
        "settings": {"total_steps": 6, "stops": [2, 4], "seed": 4},
        "sha256": {
            "corpus": sha256(corpus.read_bytes()).hexdigest(),
            "uninterrupted_checkpoint": ref,
        },
        "stages": stages,
        "checkpoint_comparison": {"step": 6, "identical": True},
    }
    (source / "multistage_audit.json").write_text(json.dumps(report), encoding="utf-8")
    return source, corpus


def test_drill_changes_only_a_copy_and_makes_hash_verification_fail(tmp_path):
    source, corpus = _fixture(tmp_path)
    original = (source / "after_step_0004.pt").read_bytes()
    drill = tmp_path / "drill"
    report = prepare_recovery_drill(source, corpus, drill, damage_step=4)
    assert report["status"] == "prepared"
    assert report["damage_step"] == 4
    assert (source / "after_step_0004.pt").read_bytes() == original
    assert (drill / "after_step_0004.pt").read_bytes() != original
    with pytest.raises(ValueError, match="Stage 4 SHA-256 mismatch"):
        verify_multistage_artifacts(drill / "multistage_audit.json", corpus)
    assert verify_multistage_artifacts(
        source / "multistage_audit.json", corpus,
    )["status"] == "verified"


def test_existing_drill_directory_is_never_overwritten(tmp_path):
    source, corpus = _fixture(tmp_path)
    destination = tmp_path / "drill"
    destination.mkdir()
    sentinel = destination / "notes.txt"
    sentinel.write_text("do not change", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_recovery_drill(source, corpus, destination, damage_step=4)
    assert sentinel.read_text(encoding="utf-8") == "do not change"


@pytest.mark.parametrize("step", [6, 3, 0, True])
def test_only_valid_intermediate_stops_can_be_damaged(tmp_path, step):
    source, corpus = _fixture(tmp_path)
    target = tmp_path / "drill"
    with pytest.raises(ValueError, match="intermediate"):
        prepare_recovery_drill(source, corpus, target, damage_step=step)
    assert not target.exists()


def test_drill_cannot_be_created_inside_original_run(tmp_path):
    source, corpus = _fixture(tmp_path)
    with pytest.raises(ValueError, match="inside"):
        prepare_recovery_drill(source, corpus, source / "drill", damage_step=2)
