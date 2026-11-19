"""Bottleneck analysis describes observations, not guaranteed speedups."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from rebuild_gpt.pipeline_analysis import summarize_pipeline_rows


def test_maximum_median_stage_and_sum_normalization():
    rows = [{
        "batch_size": 2,
        "sampling_median_ms": 1.0,
        "transfer_median_ms": 2.0,
        "update_median_ms": 7.0,
        "end_to_end_median_ms": 11.0,
    }]
    result = summarize_pipeline_rows(rows)
    assert rows[0].get("largest_median_stage") is None
    assert result[0]["largest_median_stage"] == "update"
    fractions = result[0]["stage_fraction_of_median_sum"]
    assert fractions == pytest.approx({
        "sampling": 0.1, "transfer": 0.2, "update": 0.7
    })
    assert sum(fractions.values()) == pytest.approx(1.0)
    assert "optimizer" in result[0]["candidate_investigation"]


def test_cpu_handoff_can_be_labelled_transfer_without_claiming_cuda():
    result = summarize_pipeline_rows([{
        "sampling_median_ms": "1.0",
        "transfer_median_ms": "8.0",
        "update_median_ms": "2.0",
    }])
    assert result[0]["largest_median_stage"] == "transfer"


@pytest.mark.parametrize("row", [
    {},
    {"sampling_median_ms": -1, "transfer_median_ms": 1, "update_median_ms": 1},
    {"sampling_median_ms": 0, "transfer_median_ms": 0, "update_median_ms": 0},
    {"sampling_median_ms": "nan", "transfer_median_ms": 1, "update_median_ms": 1},
    {"sampling_median_ms": 1, "transfer_median_ms": 1,
     "update_median_ms": 1, "end_to_end_median_ms": 0},
])
def test_invalid_samples_are_rejected(row):
    with pytest.raises(ValueError):
        summarize_pipeline_rows([row])


def test_report_cli_uses_real_csv_column_names(tmp_path):
    inp = tmp_path / "timing.csv"
    output = tmp_path / "analysis.json"
    with inp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "batch_size", "sampling_median_ms", "transfer_median_ms",
            "update_median_ms", "end_to_end_median_ms",
        ])
        writer.writeheader()
        writer.writerow({
            "batch_size": 2, "sampling_median_ms": 1,
            "transfer_median_ms": 2, "update_median_ms": 7,
            "end_to_end_median_ms": 10,
        })
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "examples/103_analyze_pipeline.py",
         "--input", str(inp), "--output", str(output)],
        cwd=root, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["rows"][0]["largest_median_stage"] == "update"
    assert "not additive" in data["interpretation"]
