"""Smoke-test real training-step timing for two tiny CPU shapes."""

import csv
import subprocess
import sys
from pathlib import Path


def run(*options):
    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [
            sys.executable, "examples/96_depth_batch_benchmark.py",
            "--device", "cpu", "--batch-sizes", "1,2",
            "--depths", "1", "--context", "3",
            "--n-embd", "8", "--n-head", "2",
            "--iterations", "1", "--warmup", "0",
            *options,
        ], cwd=root, capture_output=True, text=True, timeout=90,
    )


def test_benchmark_exports_two_rows_and_positive_throughput(tmp_path):
    path = tmp_path / "training.csv"
    result = run("--csv", str(path))
    assert result.returncode == 0, result.stderr
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["batch_size"]) for row in rows] == [1, 2]
    assert [int(row["tokens_per_update"]) for row in rows] == [3, 6]
    assert all(float(row["median_step_ms"]) > 0 for row in rows)
    assert all(float(row["tokens_per_second"]) > 0 for row in rows)
    assert len({int(row["parameter_count"]) for row in rows}) == 1


def test_accumulation_increases_effective_tokens(tmp_path):
    path = tmp_path / "accum.csv"
    result = run("--accumulation", "2", "--csv", str(path))
    assert result.returncode == 0, result.stderr
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["tokens_per_update"]) for row in rows] == [6, 12]


def test_invalid_head_dimension_reports_cli_error():
    result = run("--n-head", "3")
    assert result.returncode == 2
    assert "divisible" in result.stderr
