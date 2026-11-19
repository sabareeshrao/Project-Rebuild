"""Compare multiple shapes without asserting hardware-specific latency trends."""

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*flags):
    return subprocess.run(
        [
            sys.executable, "examples/102_pipeline_scaling_sweep.py",
            "--device", "cpu", "--batch-sizes", "1,2", "--depths", "1",
            "--context", "3", "--warmup", "0", "--iterations", "1",
            *flags,
        ], cwd=ROOT, capture_output=True, text=True, timeout=70,
    )


def test_cpu_sweep_writes_comparable_rows(tmp_path):
    csv_path = tmp_path / "pipeline.csv"
    json_path = tmp_path / "pipeline.json"
    result = run("--csv", str(csv_path), "--json", str(json_path))
    assert result.returncode == 0, result.stderr
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(r["batch_size"]) for r in rows] == [1, 2]
    assert [int(r["tokens_per_update"]) for r in rows] == [3, 6]
    assert len({r["parameter_count"] for r in rows}) == 1
    assert all(float(r["end_to_end_median_ms"]) > 0 for r in rows)
    assert all(r["slowest_stage"] in ("sampling", "transfer", "update") for r in rows)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["settings"]["batch_sizes"] == [1, 2]
    assert report["settings"]["depths"] == [1]
    assert report["settings"]["stage_sync"] is True
    assert report["settings"]["includes_file_io"] is False
    assert report["environment"]["device"] == "cpu"


def test_excessive_sweep_fails_before_output(tmp_path):
    path = tmp_path / "too_many.csv"
    result = run(
        "--batch-sizes", "1,2,3,4",
        "--depths", "1,2,3,4", "--csv", str(path),
    )
    assert result.returncode == 2
    assert "exceeding" in result.stderr
    assert not path.exists()


def test_context_requires_positive_length():
    result = run("--context", "0")
    assert result.returncode == 2
    assert "context_length" in result.stderr
