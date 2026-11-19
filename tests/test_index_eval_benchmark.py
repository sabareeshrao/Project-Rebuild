"""Benchmark must compare the same examples without assuming latency wins."""

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*additional):
    return subprocess.run(
        [
            sys.executable, "examples/106_index_eval_benchmark.py",
            "--device", "cpu", "--batch-size", "2", "--context", "4",
            "--eval-batches", "2", "--iterations", "1", "--warmup", "0",
            *additional,
        ], cwd=ROOT, capture_output=True, text=True, timeout=70,
    )


def test_three_backends_have_identical_loss_and_decreasing_retention(tmp_path):
    csv_file = tmp_path / "times.csv"
    json_file = tmp_path / "times.json"
    result = run("--csv", str(csv_file), "--json", str(json_file))
    assert result.returncode == 0, result.stderr
    with csv_file.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["backend"] for row in rows] == [
        "regenerate", "cached", "indices",
    ]
    assert [int(row["retained_plan_bytes"]) for row in rows] == [
        0, 2 * 2 * 2 * 4 * 2 * 8, 2 * 2 * 2 * 8,
    ]
    assert len(set(row["validation_loss"] for row in rows)) == 1
    assert all(float(row["median_evaluation_ms"]) > 0 for row in rows)
    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert data["settings"]["includes_optimizer_updates"] is False
    assert data["settings"]["includes_corpus_storage"] is False
    assert "identical validation scores" in result.stdout


def test_rejects_invalid_shape(tmp_path):
    path = tmp_path / "none.csv"
    result = run("--context", "0", "--csv", str(path))
    assert result.returncode == 2
    assert "must be positive" in result.stderr
    assert not path.exists()
