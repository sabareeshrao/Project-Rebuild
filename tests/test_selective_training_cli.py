"""Selective block checkpoint policies are preserved in resumable training."""

import json

import pytest
import torch

from rebuild_gpt.text_training import train_char_gpt
from rebuild_gpt.train_cli import main as train_main


def make_corpus(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("ab" * 140, encoding="utf-8")
    return path


def options():
    return dict(
        block_size=3, n_embd=8, n_head=2, n_layer=2,
        batch_size=3, seed=38, eval_interval=2,
        eval_batches=1, attention_impl="sdpa",
    )


def test_selective_training_matches_regular_training(tmp_path):
    corpus = make_corpus(tmp_path)
    regular, _, original_metrics = train_char_gpt(
        corpus, steps=2, **options()
    )
    selective, _, selected_metrics = train_char_gpt(
        corpus, steps=2, checkpoint_every_n=2, **options()
    )
    assert selective.blocks.checkpointed_blocks == (1,)
    assert original_metrics == pytest.approx(selected_metrics, abs=2e-6)
    for left, right in zip(regular.parameters(), selective.parameters()):
        torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-6)


def test_resume_must_use_the_same_interval(tmp_path):
    corpus = make_corpus(tmp_path)
    state = tmp_path / "training.pt"
    train_char_gpt(
        corpus, steps=2, checkpoint_every_n=2,
        training_state_path=state, **options(),
    )
    with pytest.raises(ValueError, match="controls"):
        train_char_gpt(corpus, steps=4, resume_from=state, **options())
    with pytest.raises(ValueError, match="controls"):
        train_char_gpt(
            corpus, steps=4, checkpoint_every_n=1,
            resume_from=state, **options(),
        )
    resumed, _, metrics = train_char_gpt(
        corpus, steps=4, checkpoint_every_n=2,
        resume_from=state, **options(),
    )
    assert resumed.blocks.checkpointed_blocks == (1,)
    assert metrics["train_loss_last"] > 0


def test_conflicting_and_invalid_checkpoint_options_are_rejected(tmp_path):
    corpus = make_corpus(tmp_path)
    with pytest.raises(ValueError, match="mutually exclusive"):
        train_char_gpt(
            corpus, steps=1, activation_checkpointing=True,
            checkpoint_every_n=2, **options(),
        )
    with pytest.raises(ValueError, match="every_n"):
        train_char_gpt(
            corpus, steps=1, checkpoint_every_n=3, **options(),
        )


def test_cli_reports_the_selected_interval(tmp_path):
    corpus = make_corpus(tmp_path)
    report = tmp_path / "history.json"
    args = [
        "--corpus", str(corpus), "--steps", "1", "--n-layer", "2",
        "--n-embd", "8", "--n-head", "2", "--block-size", "3",
        "--batch-size", "3", "--eval-every", "1", "--eval-batches", "1",
        "--checkpoint-every-n", "2", "--attention-impl", "sdpa",
        "--output", str(tmp_path / "final.pt"),
        "--best-output", str(tmp_path / "best.pt"),
        "--history-file", str(report),
    ]
    assert train_main(args) == 0
    saved = json.loads(report.read_text(encoding="utf-8"))
    assert saved["training_controls"]["checkpoint_every_n"] == 2
    assert saved["training_controls"]["activation_checkpointing"] is False


def test_cli_rejects_mutually_exclusive_flags(tmp_path):
    with pytest.raises(SystemExit) as exc:
        train_main([
            "--activation-checkpointing", "--checkpoint-every-n", "2",
        ])
    assert exc.value.code == 2
