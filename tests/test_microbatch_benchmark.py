"""Different chunk sizes must process the same logical training batch."""

import csv
import json
import subprocess
import sys
from pathlib import Path


def run(*args):
    root = Path(__file__).resolve().parents[1]
    return subprocess.run(
        [
            sys.executable, "examples/97_microbatch_benchmark.py",
            "--device", "cpu", "--logical-batch-size", "5",
            "--microbatch-sizes", "2,5", "--context", "3",
            "--n-embd", "8", "--n-head", "2", "--n-layer", "1",
            "--warmup", "0", "--iterations", "1", *args,
        ],
        cwd=root, capture_output=True, text=True, timeout=75,
    )


def test_fixed_logical_batch_throughput_and_cpu_memory_label(tmp_path):
    output = tmp_path / "benchmark.csv"
    json_path = tmp_path / "benchmark.json"
    result = run("--csv", str(output), "--json", str(json_path))
    assert result.returncode == 0, result.stderr
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["microbatch_size"]) for row in rows] == [2, 5]
    assert [int(row["chunks_per_step"]) for row in rows] == [3, 1]
    assert [int(row["tokens_per_update"]) for row in rows] == [15, 15]
    assert len(set(row["parameter_count"] for row in rows)) == 1
    assert all(float(row["median_step_ms"]) > 0 for row in rows)
    assert all(float(row["tokens_per_second"]) > 0 for row in rows)
    assert all(row["additional_cuda_peak_allocated_mib"] == "" for row in rows)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["settings"]["microbatch_sizes"] == [2, 5]
    assert report["environment"]["cuda_device_name"] is None
    assert all(row["additional_cuda_peak_allocated_mib"] is None
               for row in report["results"])


def test_benchmark_rejects_oversized_microbatch():
    result = run("--microbatch-sizes", "6")
    assert result.returncode == 2
    assert "cannot exceed" in result.stderr


def test_benchmark_rejects_output_path_collision(tmp_path):
    path = tmp_path / "shared"
    result = run("--csv", str(path), "--json", str(path))
    assert result.returncode == 2
    assert "distinct paths" in result.stderr
    assert not path.exists()
