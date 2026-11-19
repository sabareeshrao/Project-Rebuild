"""Transfer modes must preserve the logical workload on CPU runners."""

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(tmp_path, mode):
    csv_path = tmp_path / f"{mode}.csv"
    json_path = tmp_path / f"{mode}.json"
    args = [
        sys.executable, "examples/97_microbatch_benchmark.py",
        "--device", "cpu", "--logical-batch-size", "5",
        "--microbatch-sizes", "2,5", "--context", "3",
        "--n-embd", "8", "--n-head", "2", "--n-layer", "1",
        "--warmup", "0", "--iterations", "1",
        "--transfer-mode", mode, "--csv", str(csv_path),
        "--json", str(json_path),
    ]
    result = subprocess.run(
        args, cwd=ROOT, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stderr
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    report = json.loads(json_path.read_text(encoding="utf-8"))
    return result, rows, report


def test_eager_and_streamed_modes_use_identical_logical_workload(tmp_path):
    eager, eager_rows, eager_report = run(tmp_path, "eager")
    streamed, streamed_rows, streamed_report = run(tmp_path, "streamed")
    for rows, mode in ((eager_rows, "eager"), (streamed_rows, "streamed")):
        assert [int(row["microbatch_size"]) for row in rows] == [2, 5]
        assert [int(row["chunks_per_step"]) for row in rows] == [3, 1]
        assert [int(row["tokens_per_update"]) for row in rows] == [15, 15]
        assert all(row["source_device"] == "cpu" for row in rows)
        assert all(row["transfer_mode"] == mode for row in rows)
        assert all(row["additional_cuda_peak_allocated_mib"] == "" for row in rows)
    assert eager_report["settings"]["transfer_mode"] == "eager"
    assert streamed_report["settings"]["transfer_mode"] == "streamed"
    assert eager_report["settings"]["source_device"] == "cpu"
    assert streamed_report["environment"]["cuda_device_name"] is None
    assert "mode=streamed" in streamed.stdout
    assert "mode=eager" in eager.stdout
