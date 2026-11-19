"""The final demo must train a real model and test its saved weights."""

import pytest

from rebuild_gpt.capstone_evidence import run_review_capstone
from rebuild_gpt.capstone_spec import CapstoneSpec
from rebuild_gpt.conformance_integrity import verify_conformance_integrity


def test_real_training_to_generation_to_eight_check_evidence(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abba" * 75, encoding="utf-8")
    destination = tmp_path / "review"
    result = run_review_capstone(
        corpus, destination,
        spec=CapstoneSpec(steps=2, block_size=4, batch_size=2,
                         prompt="a", new_tokens=3, seed=7),
        conformance_seed=19,
    )
    assert result["status"] == "passed"
    assert result["checks_passed"] == 8
    assert result["capstone_integrity"]["status"] == "verified"
    assert (destination / "capstone" / "run.json").is_file()
    assert (destination / "capstone" / "sample.txt").is_file()
    assert (destination / "capstone" / "model.pt").is_file()
    assert (destination / "extended_conformance.json").is_file()
    integrity = verify_conformance_integrity(
        destination / "extended_conformance.json",
        destination / "capstone" / "model.pt", corpus,
    )
    assert integrity["status"] == "verified"


def test_existing_evidence_directory_is_not_overwritten(tmp_path):
    existing = tmp_path / "review"
    existing.mkdir()
    sentinel = existing / "notes.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        run_review_capstone(tmp_path / "missing.txt", existing)
    assert sentinel.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("seed,batch", [(-1, 2), (True, 2), (7, 1)])
def test_invalid_run_settings_fail_before_training(tmp_path, seed, batch):
    with pytest.raises(ValueError):
        run_review_capstone(
            tmp_path / "missing.txt", tmp_path / "review",
            conformance_seed=seed, batch_size=batch,
        )
    assert not (tmp_path / "review").exists()
