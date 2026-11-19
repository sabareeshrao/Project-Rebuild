"""Budget-sweep reporting supports balanced starts without changing legacy modes."""

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
    torch.manual_seed(221)
    model = GPTLanguageModel(
        GPTConfig(3, 5, 8, 2, 1, attention_impl="sdpa")
    ).eval()
    checkpoint = tmp_path / "model.pt"
    save_text_checkpoint(model, CharTokenizer("abc"), checkpoint)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abc" * 180, encoding="utf-8")
    return checkpoint, corpus


def run(checkpoint, corpus, *args):
    return subprocess.run([
        sys.executable, "examples/110_paired_budget_sweep.py",
        "--baseline", str(checkpoint), "--candidate", str(checkpoint),
        "--corpus", str(corpus), "--device", "cpu",
        "--batch-size", "2", "--context", "3",
        "--eval-batch-counts", "1,2", "--first-seed", "3",
        "--seed-count", "2", *args,
    ], cwd=ROOT, text=True, capture_output=True, timeout=100)


def test_extended_mode_includes_balanced_start_audits_and_zero_self_deltas(tmp_path):
    checkpoint, corpus = assets(tmp_path)
    csv_path, json_path = tmp_path / "budget.csv", tmp_path / "budget.json"
    result = run(
        checkpoint, corpus, "--sampling-policy", "extended",
        "--sampling-regions", "2",
        "--csv", str(csv_path), "--json", str(json_path),
    )
    assert result.returncode == 0, result.stderr
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [(row["sampling_policy"], int(row["eval_batches_per_seed"]))
            for row in rows] == [
        ("replacement", 1), ("replacement", 2),
        ("disjoint", 1), ("disjoint", 2),
        ("shifted", 1), ("shifted", 2),
        ("balanced", 1), ("balanced", 2),
    ]
    balanced = rows[-2:]
    assert [row["start_region_counts_across_seeds"] for row in balanced] == [
        "2,2", "4,4",
    ]
    assert all(row["sampling_regions"] == "2" for row in balanced)
    assert all(int(row["maximum_seed_start_imbalance"]) == 0 for row in balanced)
    assert all(int(row["seeds_with_empty_start_regions"]) == 0 for row in balanced)
    assert all(float(row["mean_candidate_minus_baseline"]) == 0 for row in rows)
    assert all(row["start_region_counts_across_seeds"] == ""
               for row in rows[:-2])
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(report["results"]) == 8
    assert report["settings"]["sampling_regions"] == 2
    assert report["settings"]["balanced_region_allocation_can_change_with_budget"]
    assert report["settings"]["sampled_windows_share_seeded_prefixes"] is False
    assert "can change with budget" in report["limitations"]


def test_original_default_and_all_still_exclude_balanced(tmp_path):
    checkpoint, corpus = assets(tmp_path)
    json_path = tmp_path / "default.json"
    assert run(checkpoint, corpus, "--json", str(json_path)).returncode == 0
    doc = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(doc["results"]) == 2
    assert doc["settings"]["sampling_policy"] == "replacement"
    assert doc["settings"]["sampled_windows_share_seeded_prefixes"] is True
    other_path = tmp_path / "all.json"
    assert run(checkpoint, corpus, "--sampling-policy", "all",
               "--json", str(other_path)).returncode == 0
    all_rows = json.loads(other_path.read_text(encoding="utf-8"))["results"]
    assert len(all_rows) == 6
    assert all(row["sampling_policy"] != "balanced" for row in all_rows)


def test_region_count_must_match_sampling_policy_and_budget(tmp_path):
    checkpoint, corpus = assets(tmp_path)
    for args in (
        ("--sampling-policy", "all", "--sampling-regions", "2"),
        ("--sampling-policy", "balanced", "--sampling-regions", "0"),
        ("--sampling-policy", "balanced", "--sampling-regions", "3"),
    ):
        target = tmp_path / "invalid.json"
        result = run(checkpoint, corpus, *args, "--json", str(target))
        assert result.returncode == 2
        assert not target.exists()
