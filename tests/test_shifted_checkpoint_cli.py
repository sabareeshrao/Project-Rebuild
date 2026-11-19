"""The shifted-policy CLI discloses actual sampled phase per seed."""

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
    torch.manual_seed(196)
    model = GPTLanguageModel(
        GPTConfig(3, 5, 8, 2, 1, attention_impl="sdpa")
    ).eval()
    checkpoint = tmp_path / "model.pt"
    save_text_checkpoint(model, CharTokenizer("abc"), checkpoint)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abc" * 170, encoding="utf-8")
    return checkpoint, corpus


def run(model, corpus, *extra):
    return subprocess.run(
        [
            sys.executable, "examples/109_compare_saved_checkpoints.py",
            "--baseline", str(model), "--candidate", str(model),
            "--corpus", str(corpus), "--device", "cpu",
            "--batch-size", "2", "--context", "4", "--eval-batches", "2",
            "--first-seed", "4", "--seed-count", "2", *extra,
        ], cwd=ROOT, capture_output=True, text=True, timeout=75,
    )


def test_shifted_policy_exports_lattice_phase_and_zero_overlap(tmp_path):
    model, corpus = assets(tmp_path)
    csv_file, json_file = tmp_path / "result.csv", tmp_path / "result.json"
    process = run(
        model, corpus, "--sampling-policy", "shifted",
        "--csv", str(csv_file), "--json", str(json_file),
    )
    assert process.returncode == 0, process.stderr
    with csv_file.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["sampling_policy"] for row in rows] == ["shifted"] * 2
    assert all(0 <= int(row["lattice_phase"]) <= 4 for row in rows)
    assert all(int(row["feasible_lattice_phases"]) >= 1 for row in rows)
    assert all(int(row["repeated_window_token_positions"]) == 0 for row in rows)
    assert all(float(row["candidate_minus_baseline"]) == 0 for row in rows)
    report = json.loads(json_file.read_text(encoding="utf-8"))
    assert report["settings"]["sampling_policy"] == "shifted"
    assert len(report["lattice_by_seed"]) == 2
    assert [int(row["lattice_phase"]) for row in rows] == [
        item["phase"] for item in report["lattice_by_seed"]
    ]
    assert report["direction_diagnostics"]["batch_directions"]["total"] == 4
    assert "not independent" in report["limitations"]


def test_default_policy_retains_no_lattice_phase(tmp_path):
    model, corpus = assets(tmp_path)
    json_file = tmp_path / "default.json"
    process = run(model, corpus, "--json", str(json_file))
    assert process.returncode == 0, process.stderr
    report = json.loads(json_file.read_text(encoding="utf-8"))
    assert report["settings"]["sampling_policy"] == "replacement"
    assert all(item["lattice"] is None for item in report["lattice_by_seed"])


def test_capacity_rejection_does_not_write_outputs(tmp_path):
    model, corpus = assets(tmp_path)
    csv_file = tmp_path / "unwritten.csv"
    process = run(
        model, corpus, "--sampling-policy", "shifted",
        "--eval-batches", "30", "--csv", str(csv_file),
    )
    assert process.returncode == 2
    assert "shifted disjoint windows require" in process.stderr
    assert not csv_file.exists()
