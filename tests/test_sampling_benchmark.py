"""Sampling benchmark is a correctness check, not a CPU speedup promise."""

import subprocess
import sys
from pathlib import Path


def test_sampling_comparison_smoke_cpu():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "examples/100_sampling_benchmark.py",
         "--batch-size", "3", "--block-size", "4",
         "--iterations", "1", "--warmup", "0"],
        cwd=root, capture_output=True, text=True, timeout=45,
    )
    assert result.returncode == 0, result.stderr
    assert "vectorized median:" in result.stdout
    assert "python_loop median:" in result.stdout
    assert "identical windows" in result.stdout


def test_sampling_comparison_rejects_empty_batch():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "examples/100_sampling_benchmark.py",
         "--batch-size", "0"],
        cwd=root, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 2
    assert "must be positive" in result.stderr
