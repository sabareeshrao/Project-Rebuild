"""Saved conformance results should identify the exact checkpoint and corpus."""

from hashlib import sha256
import json

import pytest

from rebuild_gpt.conformance_integrity import verify_conformance_integrity
from test_conformance_report import _example


def _fixture(tmp_path):
    checkpoint = tmp_path / "model.pt"
    corpus = tmp_path / "story.txt"
    checkpoint.write_bytes(b"checkpoint-contents")
    corpus.write_bytes(b"abba" * 30)
    report = _example()
    report["sha256"]["checkpoint"] = sha256(checkpoint.read_bytes()).hexdigest()
    report["sha256"]["corpus"] = sha256(corpus.read_bytes()).hexdigest()
    manifest = tmp_path / "report.json"
    manifest.write_text(json.dumps(report), encoding="utf-8")
    return manifest, checkpoint, corpus


def test_intact_files_verify_without_deserializing_pytorch(tmp_path):
    manifest, checkpoint, corpus = _fixture(tmp_path)
    result = verify_conformance_integrity(manifest, checkpoint, corpus)
    assert result["status"] == "verified"
    assert result["check_count"] == 8
    assert result["checkpoint_bytes"] == len(checkpoint.read_bytes())
    assert result["corpus_bytes"] == len(corpus.read_bytes())


def test_checkpoint_change_is_detected(tmp_path):
    manifest, checkpoint, corpus = _fixture(tmp_path)
    checkpoint.write_bytes(b"modified model")
    with pytest.raises(ValueError, match="GPT checkpoint SHA-256 mismatch"):
        verify_conformance_integrity(manifest, checkpoint, corpus)


def test_corpus_change_is_detected(tmp_path):
    manifest, checkpoint, corpus = _fixture(tmp_path)
    corpus.write_bytes(corpus.read_bytes() + b"!")
    with pytest.raises(ValueError, match="Corpus SHA-256 mismatch"):
        verify_conformance_integrity(manifest, checkpoint, corpus)


def test_missing_checkpoint_is_detected(tmp_path):
    manifest, checkpoint, corpus = _fixture(tmp_path)
    checkpoint.unlink()
    with pytest.raises(FileNotFoundError, match="GPT checkpoint"):
        verify_conformance_integrity(manifest, checkpoint, corpus)


def test_invalid_manifest_fails_before_hashing_files(tmp_path):
    manifest, checkpoint, corpus = _fixture(tmp_path)
    manifest.write_text('{"status":"passed"}', encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        verify_conformance_integrity(manifest, checkpoint, corpus)
