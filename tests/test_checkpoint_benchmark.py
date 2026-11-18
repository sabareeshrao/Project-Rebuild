"""Checkpointing performance example should run without a CUDA device."""

import subprocess
import sys
from pathlib import Path


def execute(*extra):
    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [
            sys.executable, "examples/93_activation_checkpoint_benchmark.py",
            "--device", "cpu", "--iterations", "1", "--warmup", "0",
            "--batch-size", "1", "--block-size", "3", "--n-layer", "1",
            *extra,
        ],
        cwd=root, text=True, capture_output=True, timeout=60,
    )


def test_cpu_benchmark_reports_two_modes_without_claiming_gpu_memory():
    result = execute()
    assert result.returncode == 0, result.stderr
    assert "checkpointing=off parameters=" in result.stdout
    assert "checkpointing=on parameters=" in result.stdout
    assert result.stdout.count("additional CUDA peak: unavailable (CPU)") == 2
    assert "checkpointed / regular latency ratio:" in result.stdout


def test_invalid_iterations_are_rejected():
    result = execute("--iterations", "0")
    assert result.returncode == 2
    assert "must be positive" in result.stderr


def test_cpu_fp16_is_rejected_before_running():
    result = execute("--precision", "fp16")
    assert result.returncode == 2
    assert "requires CUDA" in result.stderr
