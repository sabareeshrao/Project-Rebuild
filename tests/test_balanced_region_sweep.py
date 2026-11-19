"""Balanced sampler region count and actual token coverage are separate."""

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def make_corpus(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("abc" * 350, encoding="utf-8")
    return path


def run(source, *opts):
    return subprocess.run([
        sys.executable, "examples/113_balanced_region_sweep.py",
        "--corpus", str(source), "--batch-size", "2",
        "--context", "4", "--eval-batches", "4",
        "--first-seed", "7", "--seed-count", "2",
        "--region-counts", "2,4,8", "--position-bins", "8",
        *opts,
    ], cwd=ROOT, capture_output=True, text=True, timeout=80)


def test_region_sweep_retains_start_balance_and_reports_token_bins(tmp_path):
    source = make_corpus(tmp_path)
    csv_file, json_file = tmp_path / "sweep.csv", tmp_path / "sweep.json"
    proc = run(source, "--csv", str(csv_file), "--json", str(json_file))
    assert proc.returncode == 0, proc.stderr
    with csv_file.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["sampling_regions"]) for row in rows] == [2, 4, 8]
    assert [row["aggregate_start_counts"] for row in rows] == [
        "8,8", "4,4,4,4", "2,2,2,2,2,2,2,2",
    ]
    assert all(int(row["seeds_with_empty_start_regions"]) == 0 for row in rows)
    assert all(int(row["maximum_seed_start_imbalance"]) == 0 for row in rows)
    assert all(0 < float(row["global_token_coverage_fraction"]) <= 1
               for row in rows)
    report = json.loads(json_file.read_text(encoding="utf-8"))
    assert report["settings"]["contains_model_compute"] is False
    assert report["settings"]["heldout_token_count"] == 210
    assert len(report["regional_details"]) == 3
    for row, detail in zip(rows, report["regional_details"]):
        assert len(detail["starts_by_seed"]) == 2
        assert len(detail["token_coverage_bins"]) == 8
        assert sum(segment["unique_positions"] for segment
                   in detail["token_coverage_bins"]) == int(
                       row["global_unique_token_positions"]
                   )
        assert all(entry["empty_regions"] == 0
                   for entry in detail["starts_by_seed"])
    assert "different quantities" in report["limitations"]


def test_grid_cannot_ask_for_more_regions_than_windows(tmp_path):
    corpus = make_corpus(tmp_path)
    output = tmp_path / "invalid.json"
    result = run(corpus, "--region-counts", "2,9", "--json", str(output))
    assert result.returncode == 2
    assert "requested windows" in result.stderr
    assert not output.exists()


def test_reports_reject_overwriting_corpus_or_each_other(tmp_path):
    source = make_corpus(tmp_path)
    output = tmp_path / "same.csv"
    result = run(source, "--csv", str(output), "--json", str(output))
    assert result.returncode == 2
    assert not output.exists()
    result = run(source, "--json", str(source))
    assert result.returncode == 2
    assert source.read_text(encoding="utf-8") == "abc" * 350


def test_fixed_seeds_repeat_exactly_across_runs(tmp_path):
    source = make_corpus(tmp_path)
    first, second = tmp_path / "one.json", tmp_path / "two.json"
    assert run(source, "--json", str(first)).returncode == 0
    assert run(source, "--json", str(second)).returncode == 0
    a = json.loads(first.read_text(encoding="utf-8"))
    b = json.loads(second.read_text(encoding="utf-8"))
    assert a["results"] == b["results"]
    assert a["regional_details"] == b["regional_details"]
