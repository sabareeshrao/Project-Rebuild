"""Held-out corpus experiment must not invent model results."""

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(corpus, *flags):
    return subprocess.run([
        sys.executable, "examples/112_heldout_seed_coverage.py",
        "--corpus", str(corpus), "--batch-size", "2",
        "--context", "3", "--eval-batches", "2",
        "--seed-counts", "1,2", "--first-seed", "208", *flags,
    ], cwd=ROOT, capture_output=True, text=True, timeout=90)


def test_real_utf8_corpus_reports_global_coverage_for_all_policies(tmp_path):
    corpus = tmp_path / "story.txt"
    corpus.write_text("aébc" * 200, encoding="utf-8")
    csv_file, json_file = tmp_path / "real.csv", tmp_path / "real.json"
    proc = run(
        corpus, "--coverage-target", "0.1",
        "--csv", str(csv_file), "--json", str(json_file),
    )
    assert proc.returncode == 0, proc.stderr
    with csv_file.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    assert [int(row["seed_count"]) for row in rows] == [1, 2] * 3
    assert all(int(row["heldout_token_count"]) == 160 for row in rows)
    assert all(0 < float(row["global_corpus_coverage_fraction"]) <= 1
               for row in rows)
    report = json.loads(json_file.read_text(encoding="utf-8"))
    assert report["settings"]["corpus_token_count"] == 800
    assert report["settings"]["vocabulary_size"] == 4
    assert report["settings"]["heldout_token_count"] == 160
    assert report["settings"]["contains_model_compute"] is False
    assert report["settings"]["corpus_is_synthetic"] is False
    assert len(report["marginal_coverage"]) == 6
    assert all("coverage_target_reached" in row for row in report["results"])
    assert "not stopping rules" in report["limitations"]
    assert "held-out tokens=160" in proc.stdout


def test_empty_targets_keep_default_csv_compatible(tmp_path):
    corpus = tmp_path / "story.txt"
    corpus.write_text("abc" * 200, encoding="utf-8")
    destination = tmp_path / "baseline.csv"
    proc = run(corpus, "--sampling-policy", "replacement",
               "--csv", str(destination))
    assert proc.returncode == 0, proc.stderr
    with destination.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert "coverage_target_reached" not in rows[0]


def test_short_corpus_or_overwrite_is_rejected_before_output(tmp_path):
    short = tmp_path / "short.txt"
    short.write_text("abcabcabc", encoding="utf-8")
    dest = tmp_path / "none.csv"
    bad = run(short, "--csv", str(dest))
    assert bad.returncode == 2
    assert not dest.exists()
    large = tmp_path / "story.txt"
    large.write_text("abc" * 200, encoding="utf-8")
    overwrite = run(large, "--csv", str(large))
    assert overwrite.returncode == 2
    assert "cannot overwrite corpus" in overwrite.stderr
    assert large.read_text(encoding="utf-8") == "abc" * 200


def test_tight_disjoint_capacity_returns_error_without_csv(tmp_path):
    corpus = tmp_path / "small.txt"
    corpus.write_text("abc" * 20, encoding="utf-8")
    dest = tmp_path / "invalid.csv"
    result = run(
        corpus, "--sampling-policy", "disjoint",
        "--eval-batches", "4", "--csv", str(dest),
    )
    assert result.returncode == 2
    assert "disjoint windows require" in result.stderr
    assert not dest.exists()
