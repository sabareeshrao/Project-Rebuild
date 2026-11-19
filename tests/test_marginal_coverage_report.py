"""Seed sweep reports per-seed marginal gains without changing the union."""

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*extra):
    return subprocess.run([
        sys.executable, "examples/111_seed_coverage_sweep.py",
        "--corpus-tokens", "90", "--batch-size", "2",
        "--context", "4", "--eval-batches", "2",
        "--seed-counts", "1,2,3", "--first-seed", "205", *extra,
    ], cwd=ROOT, capture_output=True, text=True, timeout=80)


def test_marginal_traces_sum_to_global_union_for_all_policies(tmp_path):
    csv_path, json_path = tmp_path / "growth.csv", tmp_path / "growth.json"
    proc = run("--csv", str(csv_path), "--json", str(json_path))
    assert proc.returncode == 0, proc.stderr
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    report = json.loads(json_path.read_text(encoding="utf-8"))
    traces = report["marginal_coverage"]
    assert len(rows) == len(traces) == 9
    for row, trace in zip(rows, traces):
        steps = trace["steps"]
        assert len(steps) == trace["seed_count"] == int(row["seed_count"])
        assert trace["sampling_policy"] == row["sampling_policy"]
        assert [step["seed_number"] for step in steps] == list(
            range(1, len(steps) + 1)
        )
        total_new = sum(step["new_corpus_token_positions"] for step in steps)
        assert total_new == int(row["global_unique_corpus_token_positions"])
        assert steps[-1]["cumulative_unique_positions"] == total_new
        assert steps[-1]["new_corpus_token_positions"] == int(
            row["latest_seed_new_positions"]
        )
        assert sum(step["new_corpus_token_positions"] == 0 for step in steps) == (
            int(row["zero_new_coverage_seeds"])
        )
        assert all(
            step["new_corpus_token_positions"]
            + step["already_covered_positions_this_seed"]
            + step["repeated_positions_within_seed"]
            == step["window_count"] * 5 for step in steps
        )


def test_marginal_trace_is_reproducible_for_same_seeds(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    assert run("--sampling-policy", "shifted", "--json", str(a)).returncode == 0
    assert run("--sampling-policy", "shifted", "--json", str(b)).returncode == 0
    first, second = (json.loads(p.read_text(encoding="utf-8")) for p in (a, b))
    assert first["marginal_coverage"] == second["marginal_coverage"]
    assert first["results"] == second["results"]
    assert "depends on the order" in first["limitations"]
