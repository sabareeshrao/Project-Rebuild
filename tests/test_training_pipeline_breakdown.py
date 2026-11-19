"""Tiny CPU run verifies real stage instrumentation and provenance."""

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*flags):
    return subprocess.run(
        [
            sys.executable, "examples/101_training_pipeline_breakdown.py",
            "--device", "cpu", "--batch-size", "2", "--block-size", "3",
            "--n-layer", "1", "--n-embd", "8",
            "--warmup", "0", "--iterations", "1", *flags,
        ], cwd=ROOT, capture_output=True, text=True, timeout=65,
    )


def test_end_to_end_cpu_breakdown_and_json(tmp_path):
    csv_path = tmp_path / "timing.csv"
    json_path = tmp_path / "timing.json"
    result = run("--csv", str(csv_path), "--json", str(json_path))
    assert result.returncode == 0, result.stderr
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert row["mode"] == "cpu_only"
    assert row["device"] == "cpu"
    assert int(row["tokens_per_update"]) == 6
    assert all(float(row[field]) > 0 for field in (
        "sampling_median_ms", "transfer_median_ms",
        "update_median_ms", "end_to_end_median_ms",
    ))
    assert float(row["end_to_end_tokens_per_second"]) > 0
    assert row["slowest_stage"] in ("sampling", "transfer", "update")
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["settings"]["measurement"] == "sequential_stage_timing_no_overlap"
    assert report["environment"]["cuda_device_name"] is None


def test_invalid_width_fails_without_writing_csv(tmp_path):
    csv_path = tmp_path / "invalid.csv"
    result = run("--n-embd", "7", "--csv", str(csv_path))
    assert result.returncode == 2
    assert "divisible by 4" in result.stderr
    assert not csv_path.exists()
