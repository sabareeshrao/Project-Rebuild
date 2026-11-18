"""End-to-end training and CLI support for activation checkpointing."""

import json

import pytest
import torch

from rebuild_gpt.text_training import train_char_gpt
from rebuild_gpt.train_cli import main as train_main


def make_corpus(tmp_path):
    path = tmp_path / "tiny.txt"
    path.write_text("ab" * 120, encoding="utf-8")
    return path


def options():
    return dict(
        steps=2, block_size=3, n_embd=8, n_head=2, n_layer=2,
        batch_size=3, seed=99, eval_interval=1, eval_batches=1,
        attention_impl="sdpa",
    )


def test_checkpointed_training_matches_regular_cpu_run(tmp_path):
    path = make_corpus(tmp_path)
    normal, _, normal_metrics = train_char_gpt(path, **options())
    saved, _, saved_metrics = train_char_gpt(
        path, activation_checkpointing=True, **options()
    )
    assert saved.blocks.activation_checkpointing
    for p, q in zip(normal.parameters(), saved.parameters()):
        torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-6)
    for key in normal_metrics:
        assert saved_metrics[key] == pytest.approx(normal_metrics[key], abs=2e-6)


def test_resuming_requires_same_checkpointing_policy(tmp_path):
    corpus = make_corpus(tmp_path)
    state = tmp_path / "resume.pt"
    first = options()
    first["steps"] = 1
    train_char_gpt(corpus, activation_checkpointing=True,
                   training_state_path=state, **first)
    next_options = options()
    with pytest.raises(ValueError, match="controls"):
        train_char_gpt(corpus, resume_from=state, **next_options)
    resumed, _, _ = train_char_gpt(
        corpus, resume_from=state,
        activation_checkpointing=True, **next_options,
    )
    assert resumed.blocks.activation_checkpointing


def test_activation_checkpointing_flag_must_be_boolean(tmp_path):
    with pytest.raises(ValueError, match="activation_checkpointing"):
        train_char_gpt(
            make_corpus(tmp_path), activation_checkpointing=1, **options()
        )


def test_cli_exposes_checkpointing_and_records_report(tmp_path):
    corpus = make_corpus(tmp_path)
    history = tmp_path / "history.json"
    argv = [
        "--corpus", str(corpus), "--steps", "1",
        "--block-size", "3", "--n-embd", "8",
        "--n-head", "2", "--n-layer", "1", "--batch-size", "3",
        "--attention-impl", "sdpa", "--eval-every", "1",
        "--eval-batches", "1", "--activation-checkpointing",
        "--output", str(tmp_path / "final.pt"),
        "--best-output", str(tmp_path / "best.pt"),
        "--history-file", str(history),
    ]
    assert train_main(argv) == 0
    report = json.loads(history.read_text(encoding="utf-8"))
    assert report["training_controls"]["activation_checkpointing"] is True
    assert report["model_config"]["attention_impl"] == "sdpa"
