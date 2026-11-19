"""The selective benchmark must not invent GPU memory statistics on CPU."""

import subprocess
import sys
from pathlib import Path


def run(*extra):
    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [
            sys.executable, "examples/94_selective_checkpoint_benchmark.py",
            "--device", "cpu", "--iterations", "1", "--warmup", "0",
            "--batch-size", "1", "--block-size", "3", "--n-layer", "2",
            *extra,
        ],
        cwd=root, capture_output=True, text=True, timeout=75,
    )


def test_reports_none_selective_all_with_same_parameter_counts():
    result = run()
    assert result.returncode == 0, result.stderr
    assert "mode=off blocks=()" in result.stdout
    assert "mode=every_2 blocks=(1,)" in result.stdout
    assert "mode=all blocks=(0, 1)" in result.stdout
    assert result.stdout.count("additional CUDA peak: unavailable (CPU)") == 3
    assert "selective / off latency ratio:" in result.stdout
    assert "all / off latency ratio:" in result.stdout


def test_rejects_checkpoint_interval_larger_than_depth():
    result = run("--checkpoint-every-n", "3")
    assert result.returncode == 2
    assert "every_n" in result.stderr


def test_rejects_cpu_fp16():
    result = run("--precision", "fp16")
    assert result.returncode == 2
    assert "requires CUDA" in result.stderr
