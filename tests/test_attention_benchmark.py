"""Benchmark remains opt-in and does not assert hardware-dependent speed."""

import subprocess
import sys
from pathlib import Path


def test_attention_benchmark_runs_for_tiny_cpu_inputs():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "examples/88_attention_benchmark.py",
         "--device", "cpu", "--iterations", "2", "--warmup", "0",
         "--batch-size", "1", "--block-size", "3"],
        cwd=root, capture_output=True, text=True, timeout=45,
    )
    assert result.returncode == 0, result.stderr
    assert "max absolute logit difference:" in result.stdout
    assert "manual median:" in result.stdout
    assert "sdpa median:" in result.stdout
