"""One-update GPT example compares models on matched validation windows."""

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(text, *args):
    return subprocess.run(
        [
            sys.executable, "examples/108_paired_validation.py",
            "--corpus", str(text),
            "--batch-size", "2", "--context", "3",
            "--eval-batches", "2", "--first-seed", "8",
            "--seed-count", "2", *args,
        ], cwd=ROOT, capture_output=True, text=True, timeout=60,
    )


def test_paired_example_exports_before_and_after_update(tmp_path):
    text = tmp_path / "corpus.txt"
    text.write_text("abc" * 140, encoding="utf-8")
    csv_path = tmp_path / "paired.csv"
    json_path = tmp_path / "paired.json"
    proc = run(text, "--csv", str(csv_path), "--json", str(json_path))
    assert proc.returncode == 0, proc.stderr
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["seed"]) for row in rows] == [8, 9]
    assert all(float(row["baseline_loss"]) > 0 for row in rows)
    assert all(float(row["after_one_update_loss"]) > 0 for row in rows)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["settings"]["optimizer_steps_for_candidate"] == 1
    assert report["settings"]["training_step_loss"] > 0
    assert report["paired_delta_summary"]["count"] == 2
    assert "not a significance test" in report["limitations"]


def test_invalid_seed_count_reports_cli_error(tmp_path):
    text = tmp_path / "corpus.txt"
    text.write_text("abc" * 140, encoding="utf-8")
    proc = run(text, "--seed-count", "0")
    assert proc.returncode == 2
