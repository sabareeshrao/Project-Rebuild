"""Recovery CLI must distinguish inspection from actual optimizer replay."""

import json
from unittest.mock import patch

import pytest

from rebuild_gpt.resume_recovery_cli import main


def test_verify_mode_does_not_train_or_load_model(capsys):
    report = {"status": "verified", "stages": [{"step": 2}, {"step": 4}]}
    with (
        patch("rebuild_gpt.resume_recovery_cli.verify_multistage_artifacts",
              return_value=report) as verify,
        patch("rebuild_gpt.resume_recovery_cli.replay_verified_restart") as replay,
    ):
        assert main(["--run-dir", "out/run", "--corpus", "story.txt"]) == 0
    from pathlib import Path

    verify.assert_called_once_with(
        Path("out/run/multistage_audit.json"), "story.txt",
    )
    replay.assert_not_called()
    assert "Artifact integrity: verified" in capsys.readouterr().out


def test_recover_mode_saves_auditable_success_only_after_replay(tmp_path, capsys):
    directory = tmp_path / "run"
    directory.mkdir()
    manifest = directory / "multistage_audit.json"
    manifest.write_bytes(b"trusted-manifest")
    recovered = directory / "recovered.pt"
    recovered.write_bytes(b"replayed-model")
    result = {
        "status": "identical", "restart_step": 2,
        "skipped_later_stages": [],
        "recovered_checkpoint": str(recovered),
        "checkpoint_comparison": {"identical": True},
    }
    with patch(
        "rebuild_gpt.resume_recovery_cli.replay_verified_restart",
        return_value=result,
    ) as replay:
        assert main([
            "--run-dir", str(directory), "--corpus", "data/story.txt",
            "--mode", "recover",
        ]) == 0
    replay.assert_called_once_with(manifest, "data/story.txt")
    audit = json.loads((directory / "recovery_audit.json").read_text())
    assert audit["status"] == "passed"
    assert len(audit["source_manifest_sha256"]) == 64
    assert len(audit["recovered_checkpoint_sha256"]) == 64
    assert "Recovered training state" in capsys.readouterr().out


def test_failed_recovery_clears_old_success(tmp_path, capsys):
    directory = tmp_path / "run"
    directory.mkdir()
    audit = directory / "recovery_audit.json"
    audit.write_text('{"status":"passed"}')
    with patch(
        "rebuild_gpt.resume_recovery_cli.replay_verified_restart",
        side_effect=AssertionError("weights differ"),
    ):
        with pytest.raises(SystemExit) as error:
            main(["--run-dir", str(directory), "--mode", "recover"])
    assert error.value.code == 2
    assert not audit.exists()
    assert "weights differ" in capsys.readouterr().err


def test_corrupted_artifact_never_prints_verified(capsys):
    with patch(
        "rebuild_gpt.resume_recovery_cli.verify_multistage_artifacts",
        side_effect=ValueError("Stage 4 SHA-256 mismatch"),
    ):
        with pytest.raises(SystemExit) as error:
            main([])
    assert error.value.code == 2
    assert "SHA-256 mismatch" in capsys.readouterr().err


def test_unverified_replay_cannot_produce_a_passing_audit(tmp_path, capsys):
    directory = tmp_path / "run"
    directory.mkdir()
    old_audit = directory / "recovery_audit.json"
    old_audit.write_text('{"status":"passed"}', encoding="utf-8")
    with patch(
        "rebuild_gpt.resume_recovery_cli.replay_verified_restart",
        return_value={"status": "failed", "checkpoint_comparison": {"identical": False}},
    ):
        with pytest.raises(SystemExit) as error:
            main(["--run-dir", str(directory), "--mode", "recover"])
    assert error.value.code == 2
    assert not old_audit.exists()
    assert "did not match" in capsys.readouterr().err
