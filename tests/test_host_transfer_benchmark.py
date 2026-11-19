"""An honest CPU result must not invent pinned-memory speedups."""

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*flags):
    return subprocess.run(
        [
            sys.executable, "examples/99_host_transfer_benchmark.py",
            "--device", "cpu", "--batch-size", "2", "--block-size", "4",
            "--warmup", "0", "--iterations", "2", *flags,
        ],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )


def test_cpu_only_reports_sampling_with_no_fake_gpu_statistics(tmp_path):
    csv_path = tmp_path / "transfer.csv"
    json_path = tmp_path / "transfer.json"
    result = run("--csv", str(csv_path), "--json", str(json_path))
    assert result.returncode == 0, result.stderr
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["mode"] == "cpu_sampling_only"
    assert rows[0]["device"] == "cpu"
    assert int(rows[0]["sample_count"]) == 2
    assert int(rows[0]["bytes_per_two_token_batches"]) == 2 * 2 * 4 * 8
    assert float(rows[0]["median_pipeline_ms"]) > 0
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["environment"]["cuda_device_name"] is None
    assert report["settings"]["includes_model_compute"] is False
    assert "CPU reports sampling only" in result.stdout


def test_invalid_shape_rejected_without_outputs(tmp_path):
    output = tmp_path / "invalid.csv"
    result = run("--block-size", "0", "--csv", str(output))
    assert result.returncode == 2
    assert "must be positive" in result.stderr
    assert not output.exists()


def test_csv_and_json_paths_must_be_distinct(tmp_path):
    path = tmp_path / "one_output"
    result = run("--csv", str(path), "--json", str(path))
    assert result.returncode == 2
    assert "different paths" in result.stderr
    assert not path.exists()
