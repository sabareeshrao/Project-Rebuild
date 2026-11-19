"""End-to-end profile modes should report the actual selected sampler."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def execute(script, tmp_path, mode):
    output = tmp_path / (script.replace(".py", "") + "_" + mode + ".csv")
    report = tmp_path / (script.replace(".py", "") + "_" + mode + ".json")
    if script.startswith("101_"):
        args = [
            "--batch-size", "2", "--block-size", "3",
            "--n-layer", "1", "--n-embd", "8",
        ]
    else:
        args = [
            "--batch-sizes", "1,2", "--depths", "1", "--context", "3",
        ]
    result = subprocess.run(
        [
            sys.executable, "examples/" + script,
            "--device", "cpu", "--sampling-backend", mode,
            "--warmup", "0", "--iterations", "1",
            "--csv", str(output), "--json", str(report), *args,
        ],
        cwd=ROOT, text=True, capture_output=True, timeout=70,
    )
    assert result.returncode == 0, result.stderr
    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    data = json.loads(report.read_text(encoding="utf-8"))
    return rows, data


@pytest.mark.parametrize("script,expected_rows", [
    ("101_training_pipeline_breakdown.py", 1),
    ("102_pipeline_scaling_sweep.py", 2),
])
def test_indexed_and_cached_pipeline_profiles_use_same_workload(
    tmp_path, script, expected_rows,
):
    original, original_report = execute(script, tmp_path, "indexed")
    cached, cached_report = execute(script, tmp_path, "cached")
    assert len(original) == len(cached) == expected_rows
    assert original_report["settings"]["sampling_backend"] == "indexed"
    assert cached_report["settings"]["sampling_backend"] == "cached"
    assert [row["sampling_backend"] for row in original] == ["indexed"] * expected_rows
    assert [row["sampling_backend"] for row in cached] == ["cached"] * expected_rows
    for a, b in zip(original, cached):
        assert a["tokens_per_update"] == b["tokens_per_update"]
        assert a["batch_size"] == b["batch_size"]
        assert float(a["sampling_median_ms"]) > 0
        assert float(b["sampling_median_ms"]) > 0
        assert float(a["end_to_end_median_ms"]) > 0
        assert float(b["end_to_end_median_ms"]) > 0
    assert original_report["environment"]["cuda_device_name"] is None
    assert cached_report["environment"]["cuda_device_name"] is None
