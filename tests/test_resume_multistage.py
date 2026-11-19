"""Real multi-restart training must equal an uninterrupted grouped AdamW run."""

import pytest

from rebuild_gpt.resume_comparison import compare_resume_checkpoints
from rebuild_gpt.resume_multistage import compare_multistage_resume


def _corpus(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("abba" * 65, encoding="utf-8")
    return path


def test_two_real_restarts_match_all_training_state(tmp_path):
    report = compare_multistage_resume(
        _corpus(tmp_path), tmp_path / "runs",
        total_steps=6, stops=(2, 4), seed=11,
    )
    assert report["status"] == "identical"
    assert report["checkpoint_comparison"]["step"] == 6
    assert report["checkpoint_comparison"]["moment_parameters"] > 0
    assert len(report["stage_checkpoints"]) == 3
    assert report["evaluation_records"] == 7
    independent = compare_resume_checkpoints(
        report["uninterrupted_checkpoint"], report["stage_checkpoints"][-1],
    )
    assert independent["identical"]


def test_bad_schedule_is_rejected_before_writing_files(tmp_path):
    directory = tmp_path / "runs"
    with pytest.raises(ValueError, match="strictly increasing"):
        compare_multistage_resume(
            _corpus(tmp_path), directory,
            total_steps=6, stops=(4, 2),
        )
    assert not directory.exists()


def test_seed_is_validated_before_training(tmp_path):
    directory = tmp_path / "runs"
    with pytest.raises(ValueError, match="seed"):
        compare_multistage_resume(
            _corpus(tmp_path), directory, seed=-1,
        )
    assert not directory.exists()
