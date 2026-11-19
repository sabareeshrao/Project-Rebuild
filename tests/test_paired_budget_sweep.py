"""A sweep changes only sampling budget, not checkpoint weights or seeds."""

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


def data(tmp_path):
    tokenizer = CharTokenizer("abc")
    torch.manual_seed(187)
    model = GPTLanguageModel(GPTConfig(
        3, 5, 8, 2, 1, attention_impl="sdpa"
    )).eval()
    checkpoint = tmp_path / "model.pt"
    save_text_checkpoint(model, tokenizer, checkpoint)
    text = tmp_path / "corpus.txt"
    text.write_text("abc" * 150, encoding="utf-8")
    return checkpoint, text


def run(checkpoint, text, *extra):
    return subprocess.run(
        [
            sys.executable, "examples/110_paired_budget_sweep.py",
            "--baseline", str(checkpoint), "--candidate", str(checkpoint),
            "--corpus", str(text), "--device", "cpu",
            "--batch-size", "2", "--context", "3",
            "--eval-batch-counts", "1,2",
            "--first-seed", "4", "--seed-count", "2", *extra,
        ], cwd=ROOT, capture_output=True, text=True, timeout=100,
    )


def test_same_checkpoint_has_zero_deltas_for_every_budget(tmp_path):
    checkpoint, text = data(tmp_path)
    csv_path = tmp_path / "sweep.csv"
    json_path = tmp_path / "sweep.json"
    result = run(checkpoint, text, "--csv", str(csv_path), "--json", str(json_path))
    assert result.returncode == 0, result.stderr
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [int(row["eval_batches_per_seed"]) for row in rows] == [1, 2]
    assert [int(row["sampled_target_positions_per_model"]) for row in rows] == [
        1 * 2 * 2 * 3, 2 * 2 * 2 * 3,
    ]
    assert all(float(row["mean_candidate_minus_baseline"]) == 0 for row in rows)
    assert [int(row["tied_seed_means"]) for row in rows] == [2, 2]
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["settings"]["sampled_windows_share_seeded_prefixes"]
    assert report["settings"]["optimizer_updates_during_comparison"] == 0
    assert report["environment"]["device"] == "cpu"
    assert "dependent" in report["limitations"]
    assert "same seeded sampling prefix" in result.stdout


def test_oversized_count_grid_rejected_without_output(tmp_path):
    checkpoint, text = data(tmp_path)
    csv_path = tmp_path / "invalid.csv"
    result = run(
        checkpoint, text, "--eval-batch-counts", "1,2,3,4,5,6,7,8,9",
        "--csv", str(csv_path),
    )
    assert result.returncode == 2
    assert "limited" in result.stderr
    assert not csv_path.exists()


def test_output_must_not_overwrite_checkpoint(tmp_path):
    checkpoint, text = data(tmp_path)
    result = run(checkpoint, text, "--json", str(checkpoint))
    assert result.returncode == 2
    assert "cannot overwrite" in result.stderr
