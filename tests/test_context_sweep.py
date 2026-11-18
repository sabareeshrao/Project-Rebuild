"""Sweep parsing and tiny CPU performance smoke test."""

import csv
import subprocess
import sys
from pathlib import Path

import pytest

from rebuild_gpt.benchmark_sweep import parse_context_lengths


def test_context_sizes_preserve_requested_order():
    assert parse_context_lengths("16, 2,8") == (16, 2, 8)


@pytest.mark.parametrize("invalid", ["", "0", "-1", "2,2", "2,", "cat", None])
def test_context_sizes_reject_invalid_lists(invalid):
    with pytest.raises(ValueError, match="context lengths"):
        parse_context_lengths(invalid)


def test_cpu_context_sweep_produces_one_csv_row_per_length(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "report.csv"
    result = subprocess.run(
        [
            sys.executable, "examples/91_context_sweep.py",
            "--device", "cpu", "--lengths", "2,4",
            "--batch-size", "1", "--warmup", "0",
            "--iterations", "1", "--csv", str(output),
        ],
        cwd=root, text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    with output.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [int(row["context"]) for row in rows] == [2, 4]
    assert all(float(row["manual_median_ms"]) > 0 for row in rows)
    assert all(float(row["sdpa_median_ms"]) > 0 for row in rows)
    assert all(float(row["max_abs_logit_difference"]) >= 0 for row in rows)
