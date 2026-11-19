"""Sampling benchmark tests data parity and files, not timing superiority."""

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*extra):
    return subprocess.run(
        [sys.executable, "examples/104_cached_sampler_benchmark.py",
         "--batch-sizes", "2,3", "--contexts", "3",
         "--corpus-tokens", "100", "--warmup", "0", "--iterations", "1",
         *extra],
        cwd=ROOT, capture_output=True, text=True, timeout=45,
    )


def test_cpu_benchmark_checks_equivalence_and_writes_comparable_results(tmp_path):
    output = tmp_path / "sampler.csv"
    report_path = tmp_path / "sampler.json"
    result = run("--csv", str(output), "--json", str(report_path))
    assert result.returncode == 0, result.stderr
    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["batch_size"]) for row in rows] == [2, 3]
    assert all(float(row["indexed_median_ms"]) > 0 for row in rows)
    assert all(float(row["cached_median_ms"]) > 0 for row in rows)
    assert all(
        row["unfold_view_storage_bytes"] == row["corpus_storage_bytes"]
        for row in rows
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["settings"]["contains_model_compute"] is False
    assert report["environment"]["device"] == "cpu"
    assert "identical" in result.stdout


def test_invalid_context_fails_without_creating_outputs(tmp_path):
    output = tmp_path / "wrong.csv"
    result = run("--contexts", "101", "--csv", str(output))
    assert result.returncode == 2
    assert "shorter" in result.stderr
    assert not output.exists()


def test_output_paths_must_not_collide(tmp_path):
    output = tmp_path / "same_path"
    result = run("--csv", str(output), "--json", str(output))
    assert result.returncode == 2
    assert "distinct" in result.stderr
