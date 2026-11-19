"""Evaluation comparison must report the cache preparation cost honestly."""

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*flags):
    return subprocess.run(
        [
            sys.executable, "examples/105_cached_eval_benchmark.py",
            "--device", "cpu", "--eval-batches", "1,2",
            "--batch-size", "2", "--context", "3",
            "--warmup", "0", "--iterations", "1", *flags,
        ], cwd=ROOT, capture_output=True, text=True, timeout=60,
    )


def test_cpu_cache_benchmark_reports_preparation_and_repeated_evaluation(tmp_path):
    output = tmp_path / "evaluation.csv"
    json_path = tmp_path / "evaluation.json"
    proc = run("--csv", str(output), "--json", str(json_path))
    assert proc.returncode == 0, proc.stderr
    with output.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [int(row["eval_batches_per_split"]) for row in rows] == [1, 2]
    assert [int(row["plan_storage_bytes"]) for row in rows] == [
        2 * 2 * 2 * 3 * count * 8 for count in (1, 2)
    ]
    assert all(float(row["plan_preparation_median_ms"]) > 0 for row in rows)
    assert all(float(row["regenerated_eval_median_ms"]) > 0 for row in rows)
    assert all(float(row["cached_eval_median_ms"]) > 0 for row in rows)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["settings"]["measurement"] == "evaluation_only_no_optimizer"
    assert report["settings"]["preparation_reported_separately"]
    assert report["environment"]["device"] == "cpu"
    assert "speedup is not guaranteed" in proc.stdout


def test_invalid_count_fails_without_writing_report(tmp_path):
    output = tmp_path / "bad.csv"
    result = run("--eval-batches", "0", "--csv", str(output))
    assert result.returncode == 2
    assert "eval-batches" in result.stderr
    assert not output.exists()


def test_csv_json_collision_is_rejected(tmp_path):
    path = tmp_path / "same"
    result = run("--csv", str(path), "--json", str(path))
    assert result.returncode == 2
    assert "distinct paths" in result.stderr
    assert not path.exists()


def test_budgeted_benchmark_reports_break_even_and_analytic_bytes(tmp_path):
    output = tmp_path / "budget.csv"
    report = tmp_path / "budget.json"
    result = run(
        "--eval-cache-max-mib", "0.001",
        "--csv", str(output), "--json", str(report),
    )
    assert result.returncode == 0, result.stderr
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert all(row["cache_budget_mib"] == "0.001" for row in rows)
    assert all(row["estimated_net_savings_10_events_ms"] for row in rows)
    assert all(
        row["break_even_evaluation_events"] == ""
        or int(row["break_even_evaluation_events"]) >= 1
        for row in rows
    )
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["settings"]["eval_cache_max_mib"] == 0.001
    assert data["settings"]["break_even_is_estimate"] is True
    assert "Break-even calculations" in result.stdout


def test_cache_budget_rejected_before_benchmark_outputs(tmp_path):
    output = tmp_path / "too_small.csv"
    result = run(
        "--eval-cache-max-mib", "0.00001", "--csv", str(output),
    )
    assert result.returncode == 2
    assert "exceeding max_bytes" in result.stderr
    assert not output.exists()
