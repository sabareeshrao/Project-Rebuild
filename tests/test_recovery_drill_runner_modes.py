"""Verify a genuinely missing checkpoint can be replayed from a fallback."""

from unittest.mock import patch

from rebuild_gpt.multistage_resume_audit import audit_multistage_resume
from rebuild_gpt.recovery_drill_runner import run_recovery_drill
from rebuild_gpt.recovery_lab_cli import main
from rebuild_gpt.resume_health import inspect_recovery_health


def test_missing_checkpoint_recovery_matches_uninterrupted_training(tmp_path):
    corpus = tmp_path / "story.txt"
    corpus.write_text("abba" * 60, encoding="utf-8")
    original = tmp_path / "original"
    audit_multistage_resume(
        corpus, original, total_steps=4, stops=(1, 2), seed=8,
    )
    damaged = tmp_path / "damaged_copy"
    report = run_recovery_drill(
        original, corpus, damaged, damage_step=2, damage_mode="remove",
    )
    assert report["status"] == "passed"
    assert report["damage_mode"] == "remove"
    assert report["fallback_step"] == 1
    assert report["independent_check"]["status"] == "verified"
    assert not (damaged / "after_step_0002.pt").exists()
    health = inspect_recovery_health(
        original / "multistage_audit.json", corpus,
    )
    assert health["status"] == "ready"


def test_existing_cli_can_request_truncation_without_changing_defaults(capsys):
    report = {
        "status": "passed", "damage_step": 4, "damage_mode": "truncate",
        "fallback_step": 2, "independent_check": {"status": "verified"},
    }
    with patch(
        "rebuild_gpt.recovery_lab_cli.run_recovery_drill",
        return_value=report,
    ) as run:
        assert main([
            "--mode", "drill", "--source-dir", "out/original",
            "--drill-dir", "out/truncated", "--fault-mode", "truncate",
        ]) == 0
    run.assert_called_once_with(
        "out/original", "data/tiny_story.txt", "out/truncated",
        damage_step=4, damage_mode="truncate",
    )
    assert "passed" in capsys.readouterr().out
