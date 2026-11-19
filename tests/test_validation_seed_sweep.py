"""The seed sweep writes factual, deterministic per-seed losses."""

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(path, *options):
    return subprocess.run(
        [
            sys.executable, "examples/107_validation_seed_sweep.py",
            "--corpus", str(path),
            "--batch-size", "2", "--context", "3",
            "--eval-batches", "2", "--first-seed", "0",
            "--seed-count", "3", *options,
        ],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )


def test_cpu_sweep_records_model_and_sampling_seeds(tmp_path):
    text = tmp_path / "corpus.txt"
    text.write_text("abc" * 130, encoding="utf-8")
    csv_path = tmp_path / "seeds.csv"
    json_path = tmp_path / "seeds.json"
    result = run(text, "--csv", str(csv_path), "--json", str(json_path))
    assert result.returncode == 0, result.stderr
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["seed"]) for row in rows] == [0, 1, 2]
    assert all(float(row["validation_mean_loss"]) > 0 for row in rows)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["environment"]["device"] == "cpu"
    assert report["settings"]["model_is_trained"] is False
    assert report["settings"]["eval_batches_per_seed"] == 2
    assert report["summary"]["count"] == 3
    assert report["summary"]["mean"] > 0
    assert "not a calibrated" in report["limitations"]


def test_one_seed_has_null_uncertainty_values(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("abc" * 130, encoding="utf-8")
    report = tmp_path / "single.json"
    result = run(path, "--seed-count", "1", "--json", str(report))
    assert result.returncode == 0, result.stderr
    summary = json.loads(report.read_text(encoding="utf-8"))["summary"]
    assert summary["sample_stdev"] is None
    assert summary["standard_error"] is None


def test_invalid_seed_count_fails_before_output(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("abc" * 130, encoding="utf-8")
    report = tmp_path / "invalid.json"
    result = run(path, "--seed-count", "0", "--json", str(report))
    assert result.returncode == 2
    assert not report.exists()
