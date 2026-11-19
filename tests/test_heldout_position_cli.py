"""Real corpus CLI reveals positional blind spots without running a GPT."""

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def corpus(tmp_path):
    path = tmp_path / "story.txt"
    path.write_text("aébc" * 200, encoding="utf-8")
    return path


def run(source, *flags):
    return subprocess.run([
        sys.executable, "examples/112_heldout_seed_coverage.py",
        "--corpus", str(source), "--batch-size", "2",
        "--context", "3", "--eval-batches", "2",
        "--seed-counts", "1,2", "--first-seed", "212",
        *flags,
    ], cwd=ROOT, capture_output=True, text=True, timeout=100)


def test_profiles_reconcile_with_global_union_for_all_policies(tmp_path):
    source = corpus(tmp_path)
    csv_file, json_file = tmp_path / "bins.csv", tmp_path / "bins.json"
    process = run(
        source, "--position-bins", "8", "--minimum-bin-coverage", "0.5",
        "--csv", str(csv_file), "--json", str(json_file),
    )
    assert process.returncode == 0, process.stderr
    with csv_file.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    doc = json.loads(json_file.read_text(encoding="utf-8"))
    assert len(rows) == 6
    assert doc["settings"]["position_bins"] == 8
    assert doc["settings"]["minimum_bin_coverage_fraction"] == 0.5
    assert doc["settings"]["contains_model_compute"] is False
    assert len(doc["position_coverage_profiles"]) == 6
    assert len(doc["position_growth_traces"]) == 6
    assert len(doc["regional_target_assessments"]) == 6
    for row, profile, growth, target in zip(
        rows, doc["position_coverage_profiles"],
        doc["position_growth_traces"],
        doc["regional_target_assessments"],
    ):
        bins = profile["bins"]
        assert len(bins) == 8
        assert sum(segment["unique_positions"] for segment in bins) == int(
            row["global_unique_corpus_token_positions"]
        )
        assert profile["global_unique_positions"] == int(
            row["global_unique_corpus_token_positions"]
        )
        assert growth["steps"][-1]["cumulative_total_positions"] == (
            profile["global_unique_positions"]
        )
        assert bool(target["reached"]) == (row["regional_target_reached"] == "True")
        assert target["bins_below_target"] == int(row["regions_below_target"])
        assert min(item["coverage_fraction"] for item in bins) == float(
            row["least_covered_bin_fraction"]
        )


def test_position_profile_opt_in_preserves_earlier_columns(tmp_path):
    source = corpus(tmp_path)
    report = tmp_path / "original.json"
    process = run(source, "--sampling-policy", "replacement",
                  "--json", str(report))
    assert process.returncode == 0, process.stderr
    doc = json.loads(report.read_text(encoding="utf-8"))
    assert "position_coverage_profiles" not in doc
    assert "position_bins" not in doc["results"][0]


def test_invalid_bin_flags_fail_before_output(tmp_path):
    source = corpus(tmp_path)
    for flags in (
        ("--position-bins", "0"),
        ("--position-bins", "257"),
        ("--minimum-bin-coverage", "0.5"),
        ("--position-bins", "4", "--minimum-bin-coverage", "2"),
        ("--position-bins", "4", "--minimum-bin-coverage", "nan"),
    ):
        output = tmp_path / "invalid.json"
        proc = run(source, *flags, "--json", str(output))
        assert proc.returncode == 2
        assert not output.exists()
