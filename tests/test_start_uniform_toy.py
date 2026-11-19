"""An analytic constant-stratum toy checks weighted mean exactly."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run(*flags):
    return subprocess.run([
        sys.executable, "examples/115_start_uniform_toy.py",
        "--corpus-tokens", "25", "--context", "4",
        "--region-count", "4", "--batch-size", "1",
        "--eval-batches", "5", "--seed-count", "3",
        "--first-seed", "228", *flags,
    ], cwd=ROOT, capture_output=True, text=True, timeout=60)


def test_weighted_toy_recovers_known_uniform_start_population_mean(tmp_path):
    csv_path, json_path = tmp_path / "toy.csv", tmp_path / "toy.json"
    proc = run("--csv", str(csv_path), "--json", str(json_path))
    assert proc.returncode == 0, proc.stderr
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    for row in rows:
        assert row["region_start_counts"] == "2,1,1,1"
        assert float(row["exact_uniform_start_mean"]) == pytest.approx(54 / 21)
        assert float(row["raw_balanced_sample_mean"]) == pytest.approx(11 / 5)
        assert float(row["region_weighted_mean"]) == pytest.approx(54 / 21)
        assert float(row["raw_absolute_error"]) > 0.3
        assert float(row["weighted_absolute_error"]) < 1e-12
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["settings"]["contains_model_compute"] is False
    assert report["settings"]["exact_uniform_start_mean"] == pytest.approx(54 / 21)
    assert len(report["results"]) == 3
    assert "exact by construction" in report["limitations"]


def test_same_seeded_toy_is_reproducible(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    assert run("--json", str(a)).returncode == 0
    assert run("--json", str(b)).returncode == 0
    first, second = (
        json.loads(p.read_text(encoding="utf-8")) for p in (a, b)
    )
    assert first["results"] == second["results"]


def test_too_many_regions_for_budget_is_explicit_failure(tmp_path):
    output = tmp_path / "invalid.csv"
    proc = run("--region-count", "6", "--csv", str(output))
    assert proc.returncode == 2
    assert "at least one window" in proc.stderr
    assert not output.exists()


def test_outputs_must_be_distinct(tmp_path):
    output = tmp_path / "same.csv"
    proc = run("--csv", str(output), "--json", str(output))
    assert proc.returncode == 2
    assert not output.exists()
