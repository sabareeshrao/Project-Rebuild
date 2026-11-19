"""Coverage target reporting must not invent seeds or promise attainment."""

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*flags):
    return subprocess.run([
        sys.executable, "examples/111_seed_coverage_sweep.py",
        "--corpus-tokens", "90", "--batch-size", "2",
        "--context", "4", "--eval-batches", "2",
        "--seed-counts", "1,2", "--first-seed", "207",
        "--sampling-policy", "disjoint", *flags,
    ], cwd=ROOT, capture_output=True, text=True, timeout=80)


def test_reached_target_reports_first_preselected_seed(tmp_path):
    csv_file = tmp_path / "reached.csv"
    json_file = tmp_path / "reached.json"
    proc = run(
        "--coverage-target", "0.1",
        "--csv", str(csv_file), "--json", str(json_file),
    )
    assert proc.returncode == 0, proc.stderr
    with csv_file.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert all(row["coverage_target_reached"] == "True" for row in rows)
    assert all(int(row["first_reaching_seed_number"]) == 1 for row in rows)
    assert all(int(row["additional_positions_needed"]) == 0 for row in rows)
    doc = json.loads(json_file.read_text(encoding="utf-8"))
    assert doc["settings"]["coverage_target_fraction"] == 0.1
    assert len(doc["coverage_target_assessments"]) == 2
    assert all(item["reached"] is True for item in doc["coverage_target_assessments"])
    assert all(item["first_reaching_seed"] == 207
               for item in doc["coverage_target_assessments"])


def test_unattained_target_is_not_reported_as_success(tmp_path):
    path = tmp_path / "unreached.json"
    proc = run("--coverage-target", "1", "--json", str(path))
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert all(item["reached"] is False for item in doc["coverage_target_assessments"])
    assert all(item["first_reaching_seed"] is None
               for item in doc["coverage_target_assessments"])
    assert all(item["additional_positions_needed"] > 0
               for item in doc["coverage_target_assessments"])
    assert len(doc["marginal_coverage"][1]["steps"]) == 2
    assert "not statistical stopping rules" in doc["limitations"]


def test_omitted_target_keeps_legacy_rows_without_target_columns(tmp_path):
    csv_file = tmp_path / "ordinary.csv"
    proc = run("--csv", str(csv_file))
    assert proc.returncode == 0, proc.stderr
    with csv_file.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert "coverage_target_reached" not in rows[0]


def test_bad_target_fails_without_writing_outputs(tmp_path):
    for invalid in ("0", "-0.5", "1.5", "nan", "inf"):
        output = tmp_path / "invalid.json"
        proc = run("--coverage-target", invalid, "--json", str(output))
        assert proc.returncode == 2
        assert "coverage-target" in proc.stderr
        assert not output.exists()
