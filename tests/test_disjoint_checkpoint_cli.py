"""Saved-checkpoint reporting discloses overlap and sampling policy."""

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


def setup(tmp_path):
    tokenizer = CharTokenizer("abc")
    torch.manual_seed(191)
    model = GPTLanguageModel(
        GPTConfig(3, 6, 8, 2, 1, attention_impl="sdpa")
    ).eval()
    checkpoint = tmp_path / "model.pt"
    save_text_checkpoint(model, tokenizer, checkpoint)
    corpus = tmp_path / "text.txt"
    corpus.write_text("abc" * 180, encoding="utf-8")
    return checkpoint, corpus


def run(checkpoint, corpus, *flags):
    return subprocess.run(
        [
            sys.executable, "examples/109_compare_saved_checkpoints.py",
            "--baseline", str(checkpoint), "--candidate", str(checkpoint),
            "--corpus", str(corpus), "--device", "cpu",
            "--batch-size", "2", "--context", "4", "--eval-batches", "2",
            "--first-seed", "5", "--seed-count", "2", *flags,
        ],
        cwd=ROOT, capture_output=True, text=True, timeout=75,
    )


def test_disjoint_cli_retains_zero_overlap_and_model_parity(tmp_path):
    model, corpus = setup(tmp_path)
    csv_path = tmp_path / "disjoint.csv"
    json_path = tmp_path / "disjoint.json"
    result = run(
        model, corpus, "--sampling-policy", "disjoint",
        "--csv", str(csv_path), "--json", str(json_path),
    )
    assert result.returncode == 0, result.stderr
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert all(row["sampling_policy"] == "disjoint" for row in rows)
    assert all(int(row["repeated_window_token_positions"]) == 0 for row in rows)
    assert all(float(row["candidate_minus_baseline"]) == 0 for row in rows)
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["settings"]["sampling_policy"] == "disjoint"
    assert len(report["window_overlap_by_seed"]) == 2
    assert all(x["repeated_window_token_positions"] == 0
               for x in report["window_overlap_by_seed"])
    assert report["direction_diagnostics"]["batch_directions"]["total"] == 4


def test_default_policy_stays_replacement_and_reports_overlap(tmp_path):
    model, corpus = setup(tmp_path)
    json_path = tmp_path / "ordinary.json"
    result = run(model, corpus, "--json", str(json_path))
    assert result.returncode == 0, result.stderr
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["settings"]["sampling_policy"] == "replacement"
    assert len(report["window_overlap_by_seed"]) == 2


def test_disjoint_excess_sample_request_is_explicit_error(tmp_path):
    model, corpus = setup(tmp_path)
    output = tmp_path / "too_many.json"
    result = run(
        model, corpus, "--sampling-policy", "disjoint",
        "--eval-batches", "100", "--json", str(output),
    )
    assert result.returncode == 2
    assert "disjoint windows require" in result.stderr
    assert not output.exists()
