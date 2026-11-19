"""Budget sweeps must distinguish summed coverage from a global union."""

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
    torch.manual_seed(202)
    model = GPTLanguageModel(
        GPTConfig(3, 5, 8, 2, 1, attention_impl="sdpa")
    ).eval()
    checkpoint = tmp_path / "checkpoint.pt"
    save_text_checkpoint(model, CharTokenizer("abc"), checkpoint)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abc" * 160, encoding="utf-8")
    return checkpoint, corpus


def test_policies_report_deduplicated_coverage(tmp_path):
    checkpoint, corpus = fixtures(tmp_path)
    csv_file, json_file = tmp_path / "sweep.csv", tmp_path / "sweep.json"
    proc = subprocess.run([
        sys.executable, "examples/110_paired_budget_sweep.py",
        "--baseline", str(checkpoint), "--candidate", str(checkpoint),
        "--corpus", str(corpus), "--device", "cpu",
        "--batch-size", "2", "--context", "3",
        "--eval-batch-counts", "1,2", "--first-seed", "3",
        "--seed-count", "2", "--sampling-policy", "all",
        "--csv", str(csv_file), "--json", str(json_file),
    ], cwd=ROOT, capture_output=True, text=True, timeout=110)
    assert proc.returncode == 0, proc.stderr
    with csv_file.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6
    for row in rows:
        unique = int(row["global_unique_corpus_token_positions"])
        summed = int(row["total_unique_corpus_positions_across_seed_plans"])
        between = int(row["repeated_positions_between_seeds"])
        assert 0 < unique <= summed
        assert between == summed - unique
        assert 0 < float(row["global_corpus_coverage_fraction"]) <= 1
        assert int(row["duplicate_starts_between_seeds"]) >= 0
        assert float(row["mean_candidate_minus_baseline"]) == 0
    report = json.loads(json_file.read_text(encoding="utf-8"))
    assert len(report["results"]) == 6
    assert "NOT a global union" in report["limitations"]


def test_one_seed_has_no_between_seed_reuse(tmp_path):
    checkpoint, corpus = fixtures(tmp_path)
    output = tmp_path / "single.json"
    proc = subprocess.run([
        sys.executable, "examples/110_paired_budget_sweep.py",
        "--baseline", str(checkpoint), "--candidate", str(checkpoint),
        "--corpus", str(corpus), "--batch-size", "2", "--context", "3",
        "--eval-batch-counts", "2", "--seed-count", "1",
        "--json", str(output),
    ], cwd=ROOT, capture_output=True, text=True, timeout=80)
    assert proc.returncode == 0, proc.stderr
    row = json.loads(output.read_text(encoding="utf-8"))["results"][0]
    assert row["repeated_positions_between_seeds"] == 0
    assert row["duplicate_starts_between_seeds"] == 0
