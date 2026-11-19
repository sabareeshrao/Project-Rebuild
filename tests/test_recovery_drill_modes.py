"""Missing and truncated checkpoint failures are injected only into copies."""

from hashlib import sha256
import json

import pytest

from rebuild_gpt.recovery_drill import prepare_recovery_drill
from rebuild_gpt.resume_health import inspect_recovery_health


def _audited_original(tmp_path):
    run = tmp_path / "original"
    run.mkdir()
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(b"abcd" * 20)
    reference = run / "uninterrupted.pt"
    reference.write_bytes(b"reference")
    stages = []
    for step in (2, 4, 6):
        name = f"after_step_{step:04d}.pt" if step < 6 else "multistage_final.pt"
        file = run / name
        file.write_bytes(b"checkpoint-state-" + str(step).encode() * 4)
        stages.append({"step": step, "sha256": sha256(file.read_bytes()).hexdigest()})
    report = {
        "format_version": 1, "status": "passed",
        "settings": {"total_steps": 6, "stops": [2, 4], "seed": 11},
        "sha256": {
            "corpus": sha256(corpus.read_bytes()).hexdigest(),
            "uninterrupted_checkpoint": sha256(reference.read_bytes()).hexdigest(),
        },
        "stages": stages,
        "checkpoint_comparison": {"step": 6, "identical": True},
    }
    (run / "multistage_audit.json").write_text(json.dumps(report), encoding="utf-8")
    return run, corpus


@pytest.mark.parametrize("mode,expected_status", [
    ("append", "sha256_mismatch"),
    ("truncate", "sha256_mismatch"),
    ("remove", "missing"),
])
def test_each_fault_mode_preserves_original_and_falls_back(tmp_path, mode, expected_status):
    source, corpus = _audited_original(tmp_path)
    original_bytes = (source / "after_step_0004.pt").read_bytes()
    drill = tmp_path / mode
    result = prepare_recovery_drill(
        source, corpus, drill, damage_step=4, damage_mode=mode,
    )
    assert result["damage_mode"] == mode
    assert (source / "after_step_0004.pt").read_bytes() == original_bytes
    health = inspect_recovery_health(drill / "multistage_audit.json", corpus)
    assert health["status"] == "degraded"
    assert health["recommended_restart_step"] == 2
    assert health["stages"][-1]["status"] == expected_status


def test_unknown_fault_mode_rejected_before_copy(tmp_path):
    source, corpus = _audited_original(tmp_path)
    output = tmp_path / "drill"
    with pytest.raises(ValueError, match="damage_mode"):
        prepare_recovery_drill(
            source, corpus, output, damage_step=4, damage_mode="replace",
        )
    assert not output.exists()
