"""Long-format region CSV is plot-ready and reconciles with summary coverage."""

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def make_corpus(tmp_path):
    path = tmp_path / "story.txt"
    path.write_text("abc" * 250, encoding="utf-8")
    return path


def run(corpus, *flags):
    return subprocess.run([
        sys.executable, "examples/112_heldout_seed_coverage.py",
        "--corpus", str(corpus), "--context", "4",
        "--batch-size", "2", "--eval-batches", "2",
        "--seed-counts", "1,2", "--first-seed", "213",
        *flags,
    ], cwd=ROOT, text=True, capture_output=True, timeout=100)


def test_long_csv_is_one_row_per_region_per_sampling_configuration(tmp_path):
    source = make_corpus(tmp_path)
    bins_path, csv_path, json_path = (
        tmp_path / "bins.csv", tmp_path / "summary.csv", tmp_path / "profile.json",
    )
    proc = run(
        source, "--position-bins", "6", "--sampling-policy", "all",
        "--position-bin-csv", str(bins_path),
        "--csv", str(csv_path), "--json", str(json_path),
    )
    assert proc.returncode == 0, proc.stderr
    with bins_path.open(encoding="utf-8", newline="") as stream:
        bins = list(csv.DictReader(stream))
    with csv_path.open(encoding="utf-8", newline="") as stream:
        summaries = list(csv.DictReader(stream))
    assert len(bins) == 3 * 2 * 6
    assert len(summaries) == 6
    for summary in summaries:
        items = [
            row for row in bins
            if row["sampling_policy"] == summary["sampling_policy"]
            and int(row["seed_count"]) == int(summary["seed_count"])
        ]
        assert len(items) == 6
        assert [int(row["bin_index"]) for row in items] == list(range(6))
        assert sum(int(row["unique_positions"]) for row in items) == int(
            summary["global_unique_corpus_token_positions"]
        )
        assert sum(int(row["window_position_visits"]) for row in items) >= sum(
            int(row["unique_positions"]) for row in items
        )
        assert all(0 <= float(row["coverage_fraction"]) <= 1 for row in items)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(report["position_coverage_profiles"]) == 6
    assert "not stopping rules" in report["limitations"]
    assert "Position-bin CSV:" in proc.stdout


def test_long_csv_is_explicit_opt_in_and_requires_position_bins(tmp_path):
    source = make_corpus(tmp_path)
    output = tmp_path / "invalid.csv"
    proc = run(source, "--position-bin-csv", str(output))
    assert proc.returncode == 2
    assert "requires --position-bins" in proc.stderr
    assert not output.exists()


def test_outputs_cannot_overwrite_corpus_or_each_other(tmp_path):
    source = make_corpus(tmp_path)
    output = tmp_path / "duplicate.csv"
    result = run(
        source, "--position-bins", "4",
        "--position-bin-csv", str(output), "--csv", str(output),
    )
    assert result.returncode == 2
    assert not output.exists()
    result = run(
        source, "--position-bins", "4",
        "--position-bin-csv", str(source),
    )
    assert result.returncode == 2
    assert source.read_text(encoding="utf-8") == "abc" * 250
