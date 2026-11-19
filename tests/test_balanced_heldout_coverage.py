"""Four-policy held-out reports separate start balance and token coverage."""

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def corpus(tmp_path):
    path = tmp_path / "real.txt"
    path.write_text("abc" * 250, encoding="utf-8")
    return path


def run(source, *flags):
    return subprocess.run([
        sys.executable, "examples/112_heldout_seed_coverage.py",
        "--corpus", str(source), "--batch-size", "2",
        "--context", "4", "--eval-batches", "2",
        "--seed-counts", "1,2", "--first-seed", "218", *flags,
    ], cwd=ROOT, text=True, capture_output=True, timeout=120)


def test_four_policy_sweep_reports_start_balance_and_corpus_coverage(tmp_path):
    source = corpus(tmp_path)
    summary, details = tmp_path / "summary.csv", tmp_path / "details.json"
    proc = run(
        source, "--sampling-policy", "extended",
        "--sampling-regions", "4", "--position-bins", "6",
        "--csv", str(summary), "--json", str(details),
    )
    assert proc.returncode == 0, proc.stderr
    with summary.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [(row["sampling_policy"], int(row["seed_count"])) for row in rows] == [
        ("replacement", 1), ("replacement", 2),
        ("disjoint", 1), ("disjoint", 2),
        ("shifted", 1), ("shifted", 2),
        ("balanced", 1), ("balanced", 2),
    ]
    balanced = [row for row in rows if row["sampling_policy"] == "balanced"]
    assert all(row["sampling_regions"] == "4" for row in balanced)
    assert all(int(row["empty_start_regions_across_seeds"]) == 0 for row in balanced)
    assert all(int(row["maximum_start_region_imbalance"]) == 0 for row in balanced)
    assert all(0 < float(row["global_corpus_coverage_fraction"]) <= 1 for row in rows)
    report = json.loads(details.read_text(encoding="utf-8"))
    assert report["settings"]["sampling_regions"] == 4
    assert len(report["start_region_profiles"]) == 8
    assert len(report["position_coverage_profiles"]) == 8
    for entry in report["start_region_profiles"]:
        if entry["sampling_policy"] == "balanced":
            assert len(entry["seeds"]) == entry["seed_count"]
            assert all(x["counts"] == [1, 1, 1, 1] for x in entry["seeds"])
        else:
            assert entry["seeds"] == []
    assert "changes the sampling distribution" in report["limitations"]


def test_balanced_mode_reports_its_requested_region_count(tmp_path):
    source = corpus(tmp_path)
    output = tmp_path / "balanced.json"
    result = run(
        source, "--sampling-policy", "balanced",
        "--sampling-regions", "3", "--json", str(output),
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert len(report["results"]) == 2
    assert len(report["start_region_profiles"][1]["seeds"]) == 2
    assert all(x["empty_regions"] == 0
               for x in report["start_region_profiles"][1]["seeds"])


def test_original_all_option_still_selects_three_policies(tmp_path):
    source = corpus(tmp_path)
    output = tmp_path / "default.json"
    assert run(source, "--json", str(output)).returncode == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert len(report["results"]) == 6
    assert "start_region_profiles" not in report
    assert report["settings"]["sampling_policy"] == "all"


def test_region_options_must_be_valid_and_budget_adequate(tmp_path):
    source = corpus(tmp_path)
    for options in (
        ("--sampling-regions", "5"),
        ("--sampling-policy", "balanced", "--sampling-regions", "0"),
        ("--sampling-policy", "balanced", "--sampling-regions", "5"),
        ("--sampling-policy", "extended", "--sampling-regions", "5"),
    ):
        output = tmp_path / "invalid.json"
        proc = run(source, *options, "--json", str(output))
        assert proc.returncode == 2
        assert not output.exists()
