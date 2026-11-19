"""Saved checkpoint CLI reports balanced starts without changing defaults."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.text_checkpoint import save_text_checkpoint
from rebuild_gpt.tokenizer import CharTokenizer


ROOT = Path(__file__).resolve().parents[1]


def assets(tmp_path):
    torch.manual_seed(217)
    gpt = GPTLanguageModel(GPTConfig(
        3, 5, 8, 2, 1, attention_impl="sdpa",
    )).eval()
    saved = tmp_path / "gpt.pt"
    save_text_checkpoint(gpt, CharTokenizer("abc"), saved)
    corpus = tmp_path / "text.txt"
    corpus.write_text("abc" * 170, encoding="utf-8")
    return saved, corpus


def run(saved, corpus, *flags):
    return subprocess.run([
        sys.executable, "examples/109_compare_saved_checkpoints.py",
        "--baseline", str(saved), "--candidate", str(saved),
        "--corpus", str(corpus), "--device", "cpu",
        "--batch-size", "3", "--context", "4",
        "--eval-batches", "2", "--first-seed", "9",
        "--seed-count", "2", *flags,
    ], cwd=ROOT, text=True, capture_output=True, timeout=80)


def test_balanced_cli_reports_region_counts_and_identical_loss(tmp_path):
    saved, corpus = assets(tmp_path)
    csv_path, json_path = tmp_path / "balanced.csv", tmp_path / "balanced.json"
    result = run(
        saved, corpus, "--sampling-policy", "balanced",
        "--sampling-regions", "5",
        "--csv", str(csv_path), "--json", str(json_path),
    )
    assert result.returncode == 0, result.stderr
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["sampling_policy"] for row in rows] == ["balanced", "balanced"]
    assert all(row["sampling_regions"] == "5" for row in rows)
    assert all(row["start_region_counts"] == "2,1,1,1,1" for row in rows)
    assert all(int(row["empty_start_regions"]) == 0 for row in rows)
    assert all(int(row["start_region_imbalance"]) == 1 for row in rows)
    assert all(row["lattice_phase"] == "" for row in rows)
    assert all(float(row["candidate_minus_baseline"]) == 0 for row in rows)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["settings"]["sampling_policy"] == "balanced"
    assert report["settings"]["sampling_regions"] == 5
    assert len(report["start_regions_by_seed"]) == 2
    assert all(entry["counts"] == [2, 1, 1, 1, 1]
               for entry in report["start_regions_by_seed"])
    assert report["lattice_by_seed"][0]["lattice"] is None
    assert "does not ensure equal token coverage" in report["limitations"]


def test_legacy_default_remains_replacement(tmp_path):
    saved, corpus = assets(tmp_path)
    dest = tmp_path / "default.json"
    result = run(saved, corpus, "--json", str(dest))
    assert result.returncode == 0, result.stderr
    doc = json.loads(dest.read_text(encoding="utf-8"))
    assert doc["settings"]["sampling_policy"] == "replacement"
    assert doc["settings"]["sampling_regions"] is None
    assert all(entry["regions"] is None for entry in doc["start_regions_by_seed"])


def test_wrong_region_flags_are_rejected_before_outputs(tmp_path):
    saved, corpus = assets(tmp_path)
    for options in (
        ("--sampling-regions", "5"),
        ("--sampling-policy", "balanced", "--sampling-regions", "0"),
        ("--sampling-policy", "balanced", "--sampling-regions", "7"),
    ):
        dest = tmp_path / "invalid.json"
        process = run(saved, corpus, *options, "--json", str(dest))
        assert process.returncode == 2
        assert not dest.exists()


def test_balance_requires_enough_samples_per_region(tmp_path):
    saved, corpus = assets(tmp_path)
    result = run(
        saved, corpus, "--sampling-policy", "balanced",
        "--sampling-regions", "7",
    )
    assert result.returncode == 2
    assert "at least one window" in result.stderr
