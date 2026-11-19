"""Saved checkpoint CLI compares the exact vocabulary and samples."""

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


def fixtures(tmp_path):
    vocab = CharTokenizer("abc")
    torch.manual_seed(186)
    model = GPTLanguageModel(
        GPTConfig(3, 4, 8, 2, 1, attention_impl="sdpa")
    ).eval()
    baseline = tmp_path / "before.pt"
    candidate = tmp_path / "after.pt"
    save_text_checkpoint(model, vocab, baseline)
    with torch.no_grad():
        model.head.linear.weight[0, 0].add_(0.2)
    save_text_checkpoint(model, vocab, candidate)
    corpus = tmp_path / "tiny.txt"
    corpus.write_text("abc" * 150, encoding="utf-8")
    return baseline, candidate, corpus


def execute(baseline, candidate, corpus, *args):
    return subprocess.run(
        [
            sys.executable, "examples/109_compare_saved_checkpoints.py",
            "--baseline", str(baseline), "--candidate", str(candidate),
            "--corpus", str(corpus), "--device", "cpu",
            "--batch-size", "2", "--context", "3", "--eval-batches", "2",
            "--first-seed", "10", "--seed-count", "2", *args,
        ], cwd=ROOT, capture_output=True, text=True, timeout=75,
    )


def test_checkpoint_comparison_writes_seeded_csv_and_json(tmp_path):
    baseline, candidate, corpus = fixtures(tmp_path)
    csv_path = tmp_path / "comparison.csv"
    json_path = tmp_path / "comparison.json"
    result = execute(
        baseline, candidate, corpus, "--csv", str(csv_path),
        "--json", str(json_path),
    )
    assert result.returncode == 0, result.stderr
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["seed"]) for row in rows] == [10, 11]
    assert all(float(row["baseline_mean_loss"]) > 0 for row in rows)
    assert all(
        sum(int(row[field]) for field in (
            "improved_batches", "worsened_batches", "tied_batches",
        )) == 2 for row in rows
    )
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["settings"]["validation_token_count"] == 90
    assert data["settings"]["vocabulary_size"] == 3
    assert data["settings"]["optimizer_updates_during_comparison"] == 0
    assert data["paired_summary"]["count"] == 2
    assert data["direction_diagnostics"]["batch_directions"]["total"] == 4
    assert "not independent" in data["limitations"]


def test_identical_files_show_exactly_zero_delta(tmp_path):
    baseline, candidate, corpus = fixtures(tmp_path)
    result = execute(baseline, baseline, corpus)
    assert result.returncode == 0, result.stderr
    assert "paired mean delta=+0.000000" in result.stdout


def test_input_paths_are_never_overwritten(tmp_path):
    baseline, candidate, corpus = fixtures(tmp_path)
    result = execute(
        baseline, candidate, corpus,
        "--csv", str(baseline),
    )
    assert result.returncode == 2
    assert "cannot overwrite" in result.stderr
    assert baseline.exists()


def test_invalid_seed_count_rejected(tmp_path):
    baseline, candidate, corpus = fixtures(tmp_path)
    result = execute(
        baseline, candidate, corpus, "--seed-count", "0",
    )
    assert result.returncode == 2
