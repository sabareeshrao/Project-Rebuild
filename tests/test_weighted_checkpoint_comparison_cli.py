"""Checkpoint reweighting exports paired per-window loss and weights."""

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


def fixtures(tmp_path, *, changed=False, different_vocab=False):
    torch.manual_seed(227)
    tokenizer = CharTokenizer("abc")
    model = GPTLanguageModel(
        GPTConfig(3, 5, 8, 2, 1, attention_impl="sdpa")
    ).eval()
    original, candidate = tmp_path / "a.pt", tmp_path / "b.pt"
    save_text_checkpoint(model, tokenizer, original)
    if changed:
        with torch.no_grad():
            model.head.linear.weight[0, 0].add_(0.5)
    save_text_checkpoint(
        model, CharTokenizer("abd") if different_vocab else tokenizer, candidate,
    )
    corpus = tmp_path / "story.txt"
    corpus.write_text("abc" * 180, encoding="utf-8")
    return original, candidate, corpus


def run(original, candidate, corpus, *flags):
    return subprocess.run([
        sys.executable, "examples/114_weighted_checkpoint_comparison.py",
        "--baseline", str(original), "--candidate", str(candidate),
        "--corpus", str(corpus), "--device", "cpu",
        "--batch-size", "2", "--context", "4",
        "--eval-batches", "2", "--sampling-regions", "3",
        "--seed-count", "2", "--first-seed", "9", *flags,
    ], cwd=ROOT, text=True, capture_output=True, timeout=90)


def test_identical_checkpoint_exports_exact_zero_deltas(tmp_path):
    first, second, corpus = fixtures(tmp_path)
    csv_path, json_path = tmp_path / "result.csv", tmp_path / "result.json"
    result = run(first, second, corpus, "--csv", str(csv_path), "--json", str(json_path))
    assert result.returncode == 0, result.stderr
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["seed"]) for row in rows] == [9, 10]
    assert all(float(row["raw_candidate_minus_baseline"]) == 0 for row in rows)
    assert all(float(row["weighted_candidate_minus_baseline"]) == 0 for row in rows)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["settings"]["optimizer_updates_during_comparison"] == 0
    assert data["settings"]["sampling_policy"] == "balanced"
    assert len(data["region_diagnostics"]) == 2
    assert all(sum(item["region_start_counts"]) == 4
               for item in data["region_diagnostics"])
    assert "not independent" in data["limitations"]


def test_modified_checkpoint_reports_consistent_delta_signs(tmp_path):
    first, second, corpus = fixtures(tmp_path, changed=True)
    json_path = tmp_path / "changed.json"
    proc = run(first, second, corpus, "--json", str(json_path))
    assert proc.returncode == 0, proc.stderr
    rows = json.loads(json_path.read_text(encoding="utf-8"))["results"]
    assert len(rows) == 2
    assert all(abs(row["weighted_candidate_minus_baseline"] -
                   (row["weighted_candidate_loss"] - row["weighted_baseline_loss"]))
               < 1e-5 for row in rows)


def test_incompatible_checkpoint_vocab_is_rejected(tmp_path):
    first, second, corpus = fixtures(tmp_path, different_vocab=True)
    output = tmp_path / "blocked.json"
    process = run(first, second, corpus, "--json", str(output))
    assert process.returncode == 2
    assert "vocabularies must match" in process.stderr
    assert not output.exists()


def test_insufficient_samples_per_region_fail_before_output(tmp_path):
    first, second, corpus = fixtures(tmp_path)
    output = tmp_path / "invalid.csv"
    proc = run(
        first, second, corpus, "--sampling-regions", "9", "--csv", str(output),
    )
    assert proc.returncode == 2
    assert "at least one window" in proc.stderr
    assert not output.exists()
