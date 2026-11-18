"""CPU smoke checks do not fabricate a GPU memory measurement."""

import subprocess
import sys
from pathlib import Path


def run(*arguments):
    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [sys.executable, "examples/90_memory_comparison.py", *arguments],
        cwd=root, text=True, capture_output=True, timeout=60,
    )


def test_cpu_memory_comparison_reports_estimates_not_cuda_peaks():
    result = run("--device", "cpu", "--batch-size", "1", "--block-size", "3")
    assert result.returncode == 0, result.stderr
    assert "fused: trainable_parameters=" in result.stdout
    assert "sdpa: trainable_parameters=" in result.stdout
    assert result.stdout.count("estimated persistent training tensors:") == 2
    assert result.stdout.count("measured additional CUDA peak: n/a") == 2


def test_memory_comparison_rejects_zero_context_length():
    result = run("--batch-size", "1", "--block-size", "0")
    assert result.returncode == 2
    assert "must be positive" in result.stderr
