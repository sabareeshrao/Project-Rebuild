"""Do not publish successful reproduction evidence when any check fails."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from rebuild_gpt.review_reproduction import reproduce_review_evidence


def _files(tmp_path):
    folder = tmp_path / "review"
    folder.mkdir()
    (folder / "extended_conformance.json").write_bytes(b"saved-measurements")
    return folder


def _verified():
    return {
        "status": "verified", "conformance_checks": 8,
        "checkpoint_sha256": "a" * 64, "corpus_sha256": "b" * 64,
        "training_steps": 2,
    }


def _compared():
    return {
        "status": "reproduced", "reproduced_checks": 8,
        "compared_measurements": 57,
        "checkpoint_sha256": "a" * 64, "corpus_sha256": "b" * 64,
    }


def test_success_links_all_saved_evidence_and_does_not_replace_original(tmp_path):
    folder = _files(tmp_path)
    source = folder / "extended_conformance.json"
    prior = source.read_bytes()
    with (
        patch("rebuild_gpt.review_reproduction.verify_review_evidence",
              return_value=_verified()) as verify,
        patch("rebuild_gpt.review_reproduction.reproduce_conformance_report",
              return_value=_compared()) as reproduce,
    ):
        report = reproduce_review_evidence("corpus.txt", folder)
    assert report["status"] == "reproduced"
    assert report["reproduced_checks"] == 8
    assert report["compared_measurements"] == 57
    assert json.loads((folder / "reproduction.json").read_text()) == report
    assert source.read_bytes() == prior
    verify.assert_called_once_with("corpus.txt", folder)
    reproduce.assert_called_once_with(
        source, folder / "capstone" / "model.pt", "corpus.txt",
    )


def test_reproduction_failure_clears_previous_success(tmp_path):
    folder = _files(tmp_path)
    stale = folder / "reproduction.json"
    stale.write_text('{"status":"reproduced"}', encoding="utf-8")
    with (
        patch("rebuild_gpt.review_reproduction.verify_review_evidence",
              return_value=_verified()),
        patch("rebuild_gpt.review_reproduction.reproduce_conformance_report",
              side_effect=AssertionError("measured logits changed")),
    ):
        with pytest.raises(AssertionError, match="logits changed"):
            reproduce_review_evidence("corpus.txt", folder)
    assert not stale.exists()


def test_different_checkpoint_hash_is_not_a_success(tmp_path):
    folder = _files(tmp_path)
    altered = _compared()
    altered["checkpoint_sha256"] = "f" * 64
    with (
        patch("rebuild_gpt.review_reproduction.verify_review_evidence",
              return_value=_verified()),
        patch("rebuild_gpt.review_reproduction.reproduce_conformance_report",
              return_value=altered),
    ):
        with pytest.raises(AssertionError, match="different checkpoint"):
            reproduce_review_evidence("corpus.txt", folder)
    assert not (folder / "reproduction.json").exists()


def test_incomplete_reproduction_is_never_accepted(tmp_path):
    folder = _files(tmp_path)
    incomplete = _compared()
    incomplete["reproduced_checks"] = 7
    with (
        patch("rebuild_gpt.review_reproduction.verify_review_evidence",
              return_value=_verified()),
        patch("rebuild_gpt.review_reproduction.reproduce_conformance_report",
              return_value=incomplete),
    ):
        with pytest.raises(AssertionError, match="All eight"):
            reproduce_review_evidence("corpus.txt", folder)
