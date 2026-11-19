"""Checkpoint JSON includes corpus-wide token coverage across seeds."""

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


def make_assets(tmp_path):
    tokenizer = CharTokenizer("abc")
    torch.manual_seed(201)
    model = GPTLanguageModel(
        GPTConfig(3, 5, 8, 2, 1, attention_impl="sdpa")
    ).eval()
    checkpoint = tmp_path / "model.pt"
    save_text_checkpoint(model, tokenizer, checkpoint)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abc" * 160, encoding="utf-8")
    return checkpoint, corpus


def run(model, corpus, *extra):
    return subprocess.run([
        sys.executable, "examples/109_compare_saved_checkpoints.py",
        "--baseline", str(model), "--candidate", str(model),
        "--corpus", str(corpus), "--device", "cpu",
        "--batch-size", "2", "--context", "4", "--eval-batches", "2",
        "--first-seed", "7", "--seed-count", "2", *extra,
    ], cwd=ROOT, capture_output=True, text=True, timeout=90)


def test_saved_checkpoint_json_contains_global_union_and_repeat_accounting(tmp_path):
    model, corpus = make_assets(tmp_path)
    path = tmp_path / "report.json"
    proc = run(model, corpus, "--sampling-policy", "disjoint",
               "--json", str(path))
    assert proc.returncode == 0, proc.stderr
    report = json.loads(path.read_text(encoding="utf-8"))
    global_stats = report["cross_seed_overlap"]
    assert global_stats["seed_count"] == 2
    assert global_stats["window_count"] == 8
    assert global_stats["repeated_positions_within_seeds"] == 0
    assert global_stats["repeated_positions_between_seeds"] >= 0
    assert global_stats["total_repeated_positions"] == (
        global_stats["total_window_token_positions"]
        - global_stats["unique_corpus_token_positions"]
    )
    assert 0 < global_stats["coverage_fraction"] <= 1
    assert len(report["window_overlap_by_seed"]) == 2
    assert "global unique token positions" in proc.stdout
    assert "not independent" in report["limitations"]


def test_one_seed_reports_zero_between_seed_repetition(tmp_path):
    model, corpus = make_assets(tmp_path)
    path = tmp_path / "one_seed.json"
    proc = run(
        model, corpus, "--seed-count", "1", "--json", str(path),
    )
    assert proc.returncode == 0, proc.stderr
    coverage = json.loads(path.read_text(encoding="utf-8"))["cross_seed_overlap"]
    assert coverage["seed_count"] == 1
    assert coverage["duplicate_starts_between_seeds"] == 0
    assert coverage["repeated_positions_between_seeds"] == 0
