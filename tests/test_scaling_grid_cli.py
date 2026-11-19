"""Scaling grid exports stable CSV without allocating a GPT."""

import csv
import subprocess
import sys
from pathlib import Path


def call(*args):
    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, "examples/95_scaling_grid.py", *args],
        cwd=root, capture_output=True, text=True, timeout=45,
    )


def test_grid_csv_values_and_order(tmp_path):
    path = tmp_path / "grid.csv"
    result = call(
        "--batch-sizes", "1,2",
        "--contexts", "4",
        "--depths", "1,3",
        "--heads", "2",
        "--accumulation", "2",
        "--csv", str(path),
    )
    assert result.returncode == 0, result.stderr
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(int(r["batch_size"]), int(r["n_layer"])) for r in rows] == [
        (1, 1), (1, 3), (2, 1), (2, 3),
    ]
    assert int(rows[0]["tokens_per_optimizer_step"]) == 8
    assert int(rows[0]["explicit_score_bytes_per_layer"]) == 1 * 2 * 4 * 4 * 4
    assert int(rows[1]["sum_explicit_score_bytes_all_layers"]) == 3 * 128
    assert "analytical" in result.stdout


def test_bf16_changes_estimate_without_requiring_cuda(tmp_path):
    out = tmp_path / "bf16.csv"
    result = call(
        "--batch-sizes", "1", "--contexts", "4", "--depths", "1",
        "--heads", "2", "--dtype", "bf16", "--csv", str(out),
    )
    assert result.returncode == 0, result.stderr
    with out.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert int(row["explicit_score_bytes_per_layer"]) == 1 * 2 * 4 * 4 * 2


def test_grid_limit_fails_before_export(tmp_path):
    out = tmp_path / "too_many.csv"
    result = call(
        "--batch-sizes", "1,2,3", "--contexts", "4,8",
        "--depths", "1,2", "--max-points", "8", "--csv", str(out),
    )
    assert result.returncode == 2
    assert "exceeding" in result.stderr
    assert not out.exists()
