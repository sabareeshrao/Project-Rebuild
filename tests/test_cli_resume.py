"""Verify that resumed CLI training matches uninterrupted CLI training."""

import json

import torch

from rebuild_gpt.text_checkpoint import load_text_checkpoint
from rebuild_gpt.train_cli import main as train_main


def make_options(tmp_path):
    corpus = tmp_path / "repeat.txt"
    corpus.write_text("ab" * 140, encoding="utf-8")
    return [
        "--corpus", str(corpus), "--steps", "4",
        "--block-size", "3", "--n-embd", "8", "--n-head", "2",
        "--n-layer", "1", "--batch-size", "3", "--seed", "15",
        "--eval-every", "2", "--eval-batches", "2",
        "--grouped-adamw", "--weight-decay", "0.1",
        "--gradient-accumulation-steps", "2",
        "--warmup-steps", "1", "--lr-decay-steps", "4",
        "--min-lr", "0.002", "--grad-clip", "1.0",
        "--restore-best",
        "--output", str(tmp_path / "final.pt"),
        "--best-output", str(tmp_path / "best.pt"),
        "--history-file", str(tmp_path / "history.json"),
    ]


def test_cli_resumes_grouped_adamw_training_with_identical_results(tmp_path):
    resumed_dir = tmp_path / "resumed"
    resumed_dir.mkdir()
    complete_dir = tmp_path / "complete"
    complete_dir.mkdir()

    resumed_opts = make_options(resumed_dir)
    resumed_opts[resumed_opts.index("--steps") + 1] = "2"
    state_path = resumed_dir / "train_state.pt"
    assert train_main(resumed_opts + ["--training-state", str(state_path)]) == 0

    resumed_opts[resumed_opts.index("--steps") + 1] = "4"
    assert train_main(resumed_opts + [
        "--resume-from", str(state_path), "--training-state", str(state_path),
    ]) == 0

    complete_opts = make_options(complete_dir)
    assert train_main(complete_opts) == 0

    resumed_report = json.loads((resumed_dir / "history.json").read_text())
    complete_report = json.loads((complete_dir / "history.json").read_text())
    assert resumed_report["evaluations"] == complete_report["evaluations"]
    assert resumed_report["metrics"] == complete_report["metrics"]
    assert resumed_report["training_controls"] == complete_report["training_controls"]
    assert resumed_report["resumed_from"] == str(state_path)
    resumed_model, resumed_tokenizer = load_text_checkpoint(resumed_dir / "final.pt")
    complete_model, complete_tokenizer = load_text_checkpoint(complete_dir / "final.pt")
    assert resumed_tokenizer.vocab == complete_tokenizer.vocab
    assert all(
        torch.equal(p, q) for p, q in
        zip(resumed_model.parameters(), complete_model.parameters())
    )
