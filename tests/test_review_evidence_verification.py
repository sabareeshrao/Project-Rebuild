"""Cross-check two trusted manifests before drawing any final conclusion."""

from hashlib import sha256
import json

import pytest

from rebuild_gpt.review_evidence_verification import verify_review_evidence
from test_conformance_report import _example


def _bundle(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_bytes(b"abba" * 45)
    output = tmp_path / "review"
    training = output / "capstone"
    training.mkdir(parents=True)
    (training / "model.pt").write_bytes(b"tiny-gpt-checkpoint")
    (training / "sample.txt").write_text("aabb", encoding="utf-8")
    corpus_hash = sha256(corpus.read_bytes()).hexdigest()
    model_hash = sha256((training / "model.pt").read_bytes()).hexdigest()
    run = {
        "format_version": 1, "checkpoint": "model.pt",
        "settings": {"seed": 7, "steps": 2, "block_size": 4, "n_layer": 1},
        "metrics": {
            "validation_before": 2.0,
            "validation_after": 1.8,
            "train_loss_last": 1.7,
        },
        "generated_text": "aabb",
        "corpus_sha256": corpus_hash,
        "checkpoint_sha256": model_hash,
    }
    (training / "run.json").write_text(json.dumps(run), encoding="utf-8")
    conformance = _example()
    conformance["sha256"] = {
        "corpus": corpus_hash, "checkpoint": model_hash,
    }
    (output / "extended_conformance.json").write_text(
        json.dumps(conformance), encoding="utf-8",
    )
    return corpus, output


def test_both_manifests_match_actual_bytes(tmp_path):
    corpus, output = _bundle(tmp_path)
    report = verify_review_evidence(corpus, output)
    assert report["status"] == "verified"
    assert report["conformance_checks"] == 8
    assert report["training_steps"] == 2


def test_conformance_digest_cannot_claim_different_weights(tmp_path):
    corpus, output = _bundle(tmp_path)
    path = output / "extended_conformance.json"
    info = json.loads(path.read_text())
    info["sha256"]["checkpoint"] = "a" * 64
    path.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_review_evidence(corpus, output)


def test_same_file_bytes_do_not_excuse_mismatched_model_configuration(tmp_path):
    corpus, output = _bundle(tmp_path)
    path = output / "extended_conformance.json"
    info = json.loads(path.read_text())
    info["model"]["n_layer"] = 2
    path.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="n_layer"):
        verify_review_evidence(corpus, output)


def test_nonfinite_validation_metric_cannot_support_release(tmp_path):
    corpus, output = _bundle(tmp_path)
    path = output / "capstone" / "run.json"
    info = json.loads(path.read_text())
    info["metrics"]["validation_after"] = float("nan")
    path.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError, match="validation_after"):
        verify_review_evidence(corpus, output)
