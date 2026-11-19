"""Hash-only capstone verification must reject altered artifacts."""

from hashlib import sha256
import json

import pytest

from rebuild_gpt.capstone_integrity import verify_capstone_artifacts


def _files(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    corpus = tmp_path / "story.txt"
    corpus.write_bytes(b"abba" * 40)
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "model.pt").write_bytes(b"trusted GPT checkpoint bytes")
    (folder / "sample.txt").write_text("abba", encoding="utf-8")
    manifest = {
        "format_version": 1, "checkpoint": "model.pt",
        "settings": {"seed": 7}, "metrics": {"validation_after": 1.2},
        "generated_text": "abba",
        "corpus_sha256": sha256(corpus.read_bytes()).hexdigest(),
        "checkpoint_sha256": sha256((folder / "model.pt").read_bytes()).hexdigest(),
    }
    (folder / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
    return corpus, folder


def test_valid_files_verify_without_loading_torch(tmp_path):
    corpus, folder = _files(tmp_path)
    result = verify_capstone_artifacts(corpus, folder)
    assert result["status"] == "verified"
    assert len(result["corpus_sha256"]) == 64
    assert result["sample_bytes"] == 4


def test_changed_checkpoint_rejected(tmp_path):
    corpus, folder = _files(tmp_path)
    (folder / "model.pt").write_bytes(b"modified model")
    with pytest.raises(ValueError, match="Capstone model SHA-256 mismatch"):
        verify_capstone_artifacts(corpus, folder)


def test_changed_sample_rejected(tmp_path):
    corpus, folder = _files(tmp_path)
    (folder / "sample.txt").write_text("other", encoding="utf-8")
    with pytest.raises(ValueError, match="sample differs"):
        verify_capstone_artifacts(corpus, folder)


def test_reject_different_corpus_and_missing_checkpoint(tmp_path):
    corpus, folder = _files(tmp_path)
    corpus.write_bytes(b"new text")
    with pytest.raises(ValueError, match="Capstone corpus SHA-256 mismatch"):
        verify_capstone_artifacts(corpus, folder)
    corpus, folder = _files(tmp_path.parent / "second")
    (folder / "model.pt").unlink()
    with pytest.raises(FileNotFoundError, match="Capstone model"):
        verify_capstone_artifacts(corpus, folder)


def test_reject_malformed_digest_before_reading_files(tmp_path):
    corpus, folder = _files(tmp_path)
    path = folder / "run.json"
    report = json.loads(path.read_text())
    report["checkpoint_sha256"] = "not-a-digest"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Malformed"):
        verify_capstone_artifacts(corpus, folder)
