"""Model-free coverage sweeps preserve seeded prefix and honest statistics."""

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*flags):
    return subprocess.run(
        [
            sys.executable, "examples/111_seed_coverage_sweep.py",
            "--corpus-tokens", "90", "--batch-size", "2",
            "--context", "4", "--eval-batches", "2",
            "--seed-counts", "1,2", "--first-seed", "203", *flags,
        ], cwd=ROOT, text=True, capture_output=True, timeout=60,
    )


def test_all_policies_show_monotone_global_coverage_and_repeated_positions(tmp_path):
    csv_file, json_file = tmp_path / "coverage.csv", tmp_path / "coverage.json"
    result = run("--csv", str(csv_file), "--json", str(json_file))
    assert result.returncode == 0, result.stderr
    with csv_file.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [(row["sampling_policy"], int(row["seed_count"])) for row in rows] == [
        ("replacement", 1), ("replacement", 2),
        ("disjoint", 1), ("disjoint", 2),
        ("shifted", 1), ("shifted", 2),
    ]
    for first, second in zip(rows[::2], rows[1::2]):
        assert first["sampling_policy"] == second["sampling_policy"]
        one = int(first["global_unique_corpus_token_positions"])
        two = int(second["global_unique_corpus_token_positions"])
        assert one <= two <= 90
        assert int(first["repeated_positions_between_seeds"]) == 0
        assert int(second["repeated_positions_between_seeds"]) >= 0
        assert 0 < float(second["global_corpus_coverage_fraction"]) <= 1
    assert all(
        int(row["repeated_positions_within_seeds"]) == 0
        for row in rows if row["sampling_policy"] != "replacement"
    )
    report = json.loads(json_file.read_text(encoding="utf-8"))
    assert report["settings"]["contains_model_compute"] is False
    assert report["settings"]["corpus_is_synthetic"] is True
    assert len(report["results"]) == 6
    assert "does not measure" in report["limitations"]


def test_oversized_disjoint_request_fails_before_output(tmp_path):
    csv_file = tmp_path / "invalid.csv"
    proc = run(
        "--sampling-policy", "disjoint",
        "--corpus-tokens", "19", "--csv", str(csv_file),
    )
    assert proc.returncode == 2
    assert "only 3" in proc.stderr
    assert not csv_file.exists()


def test_invalid_seed_count_grid_rejected(tmp_path):
    json_file = tmp_path / "invalid.json"
    proc = run("--seed-counts", "0,2", "--json", str(json_file))
    assert proc.returncode == 2
    assert "seed-counts" in proc.stderr
    assert not json_file.exists()
