"""Compare all three evaluation policies across budgets and disclose phase."""

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


def corpus_and_checkpoint(tmp_path):
    torch.manual_seed(197)
    gpt = GPTLanguageModel(GPTConfig(
        3, 5, 8, 2, 1, attention_impl="sdpa",
    )).eval()
    checkpoint = tmp_path / "gpt.pt"
    save_text_checkpoint(gpt, CharTokenizer("abc"), checkpoint)
    corpus = tmp_path / "text.txt"
    corpus.write_text("abc" * 180, encoding="utf-8")
    return checkpoint, corpus


def run(checkpoint, corpus, *flags):
    return subprocess.run(
        [
            sys.executable, "examples/110_paired_budget_sweep.py",
            "--baseline", str(checkpoint), "--candidate", str(checkpoint),
            "--corpus", str(corpus), "--device", "cpu",
            "--batch-size", "2", "--context", "3",
            "--eval-batch-counts", "1,2", "--first-seed", "7",
            "--seed-count", "2", *flags,
        ], cwd=ROOT, capture_output=True, text=True, timeout=100,
    )


def test_all_policies_have_identical_self_comparisons_and_phase_provenance(tmp_path):
    model, corpus = corpus_and_checkpoint(tmp_path)
    csv_file, json_file = tmp_path / "all.csv", tmp_path / "all.json"
    process = run(
        model, corpus, "--sampling-policy", "all",
        "--csv", str(csv_file), "--json", str(json_file),
    )
    assert process.returncode == 0, process.stderr
    with csv_file.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [(row["sampling_policy"], int(row["eval_batches_per_seed"]))
            for row in rows] == [
        ("replacement", 1), ("replacement", 2),
        ("disjoint", 1), ("disjoint", 2),
        ("shifted", 1), ("shifted", 2),
    ]
    assert all(float(row["mean_candidate_minus_baseline"]) == 0 for row in rows)
    assert all(int(row["tied_seed_means"]) == 2 for row in rows)
    assert all(
        row["lattice_phases_by_seed"] == "n/a,n/a"
        for row in rows if row["sampling_policy"] == "replacement"
    )
    assert all(
        row["lattice_phases_by_seed"] == "0,0"
        for row in rows if row["sampling_policy"] == "disjoint"
    )
    assert all(
        len(row["lattice_phases_by_seed"].split(",")) == 2
        for row in rows if row["sampling_policy"] == "shifted"
    )
    assert all(
        int(row["repeated_window_positions_across_seeds"]) == 0
        for row in rows if row["sampling_policy"] != "replacement"
    )
    report = json.loads(json_file.read_text(encoding="utf-8"))
    assert report["settings"]["sampling_policy"] == "all"
    assert report["settings"]["sampled_windows_share_seeded_prefixes"] is False
    assert report["settings"]["shifted_phase_set_can_change_with_budget"] is True
    assert "need not share prefixes" in report["limitations"]
    assert "policy=shifted" in process.stdout


def test_original_both_option_keeps_prefix_guarantee(tmp_path):
    model, corpus = corpus_and_checkpoint(tmp_path)
    json_file = tmp_path / "both.json"
    proc = run(
        model, corpus, "--sampling-policy", "both",
        "--json", str(json_file),
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(json_file.read_text(encoding="utf-8"))
    assert len(report["results"]) == 4
    assert report["settings"]["sampled_windows_share_seeded_prefixes"] is True


def test_shifted_over_capacity_rejects_without_output(tmp_path):
    model, corpus = corpus_and_checkpoint(tmp_path)
    csv_file = tmp_path / "too_big.csv"
    proc = run(
        model, corpus, "--sampling-policy", "shifted",
        "--eval-batch-counts", "30", "--csv", str(csv_file),
    )
    assert proc.returncode == 2
    assert "shifted disjoint windows require" in proc.stderr
    assert not csv_file.exists()
