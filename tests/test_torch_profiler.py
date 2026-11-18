"""Smoke-test opt-in profiling without requiring CUDA."""

import json
import subprocess
import sys
from pathlib import Path


def run(tmp_path, *extra):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "gpt_trace.json"
    result = subprocess.run(
        [
            sys.executable, "examples/92_torch_profiler.py",
            "--device", "cpu", "--steps", "1", "--batch-size", "1",
            "--block-size", "3", "--output", str(output), *extra,
        ],
        cwd=root, text=True, capture_output=True, timeout=60,
    )
    return result, output


def test_cpu_profile_exports_chrome_trace_with_events(tmp_path):
    result, output = run(tmp_path)
    assert result.returncode == 0, result.stderr
    assert "Trace:" in result.stdout
    assert output.exists()
    document = json.loads(output.read_text(encoding="utf-8"))
    assert isinstance(document.get("traceEvents"), list)
    assert document["traceEvents"]


def test_profile_requires_positive_step_count(tmp_path):
    result, output = run(tmp_path, "--steps", "0")
    assert result.returncode == 2
    assert "must be positive" in result.stderr
    assert not output.exists()
