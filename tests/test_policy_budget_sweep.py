"""Compare both policies without confusing overlap with significance."""

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


def prepare(tmp_path):
    torch.manual_seed(192)
    model = GPTLanguageModel(
        GPTConfig(3, 5, 8, 2, 1, attention_impl="sdpa")
    ).eval()
    checkpoint = tmp_path / "gpt.pt"
    save_text_checkpoint(model, CharTokenizer("abc"), checkpoint)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abc" * 180, encoding="utf-8")
    return checkpoint, corpus


def run(checkpoint, corpus, *additional):
    return subprocess.run(
        [
            sys.executable, "examples/110_paired_budget_sweep.py",
            "--baseline", str(checkpoint), "--candidate", str(checkpoint),
            "--corpus", str(corpus), "--device", "cpu",
            "--batch-size", "2", "--context", "3",
            "--first-seed", "7", "--seed-count", "2",
            "--eval-batch-counts", "1,2", *additional,
        ], cwd=ROOT, capture_output=True, text=True, timeout=100,
    )


def test_both_policies_match_identical_checkpoint_and_disclose_overlap(tmp_path):
    model, corpus = prepare(tmp_path)
    csv_file = tmp_path / "policies.csv"
    json_file = tmp_path / "policies.json"
    result = run(
        model, corpus, "--sampling-policy", "both",
        "--csv", str(csv_file), "--json", str(json_file),
    )
    assert result.returncode == 0, result.stderr
    with csv_file.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [(row["sampling_policy"], int(row["eval_batches_per_seed"]))
            for row in rows] == [
        ("replacement", 1), ("replacement", 2),
        ("disjoint", 1), ("disjoint", 2),
    ]
    assert all(float(row["mean_candidate_minus_baseline"]) == 0 for row in rows)
    assert all(int(row["tied_seed_means"]) == 2 for row in rows)
    assert all(
        int(row["repeated_window_positions_across_seeds"]) == 0
        for row in rows if row["sampling_policy"] == "disjoint"
    )
    report = json.loads(json_file.read_text(encoding="utf-8"))
    assert report["settings"]["sampling_policy"] == "both"
    assert report["settings"]["optimizer_updates_during_comparison"] == 0
    assert "different distributions" in report["limitations"]
    assert "policy=disjoint" in result.stdout


def test_disjoint_capacity_error_does_not_write_csv(tmp_path):
    model, corpus = prepare(tmp_path)
    csv_file = tmp_path / "too_many.csv"
    result = run(
        model, corpus, "--sampling-policy", "disjoint",
        "--eval-batch-counts", "40", "--csv", str(csv_file),
    )
    assert result.returncode == 2
    assert "disjoint windows require" in result.stderr
    assert not csv_file.exists()


def test_default_sweep_remains_replacement(tmp_path):
    model, corpus = prepare(tmp_path)
    csv_file = tmp_path / "default.csv"
    result = run(model, corpus, "--csv", str(csv_file))
    assert result.returncode == 0, result.stderr
    with csv_file.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert all(row["sampling_policy"] == "replacement" for row in rows)
