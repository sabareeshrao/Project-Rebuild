"""The existing trainer must resume a real grouped AdamW run exactly."""

import pytest

from rebuild_gpt.resume_equivalence import run_resume_equivalence


def _corpus(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("abba" * 55, encoding="utf-8")
    return path


def test_split_and_uninterrupted_cpu_training_match_exactly(tmp_path):
    report = run_resume_equivalence(
        _corpus(tmp_path), tmp_path / "run",
        first_steps=2, total_steps=4, seed=19,
    )
    assert report["status"] == "identical"
    assert report["checkpoint_comparison"]["identical"]
    assert report["checkpoint_comparison"]["step"] == 4
    assert report["checkpoint_comparison"]["moment_parameters"] > 0
    assert report["evaluation_records"] == 5
    assert (tmp_path / "run" / "full.pt").is_file()
    assert (tmp_path / "run" / "partial.pt").is_file()
    assert (tmp_path / "run" / "resumed.pt").is_file()


@pytest.mark.parametrize("first,total,seed", [
    (0, 4, 1), (4, 4, 1), (5, 4, 1),
    (True, 4, 1), (1, 3, -2),
])
def test_bad_resume_experiment_settings_rejected(tmp_path, first, total, seed):
    with pytest.raises(ValueError):
        run_resume_equivalence(
            _corpus(tmp_path), tmp_path / "run",
            first_steps=first, total_steps=total, seed=seed,
        )
