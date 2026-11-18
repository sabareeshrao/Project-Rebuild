"""Training performance measurement is opt-in and has no speedup assertion."""

import subprocess
import sys
from pathlib import Path


def test_training_step_benchmark_smoke_cpu():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable, "examples/89_training_benchmark.py",
            "--device", "cpu", "--iterations", "1", "--warmup", "0",
            "--batch-size", "1", "--block-size", "3",
        ],
        cwd=root, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "tokens per optimizer step: 3" in result.stdout
    assert "initial max absolute logit difference:" in result.stdout
    assert "manual training-step median:" in result.stdout
    assert "sdpa training-step median:" in result.stdout
    assert "manual tokens/sec:" in result.stdout
    assert "sdpa tokens/sec:" in result.stdout
