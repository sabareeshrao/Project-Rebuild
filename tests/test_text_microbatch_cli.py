"""Real character training can split one logical batch without changing data."""

import json

import pytest
import torch

from rebuild_gpt.text_training import train_char_gpt
from rebuild_gpt.train_cli import main as train_main


def corpus(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("ab" * 140, encoding="utf-8")
    return path


def options():
    return dict(
        steps=2, block_size=3, n_embd=8, n_head=2, n_layer=1,
        batch_size=5, seed=140, learning_rate=0.001,
        eval_interval=1, eval_batches=1, attention_impl="sdpa",
    )


@pytest.mark.parametrize("micro_size", [1, 2, 3, 5])
def test_chunked_training_matches_unsplit_logical_batch(tmp_path, micro_size):
    path = corpus(tmp_path)
    full, _, full_metrics = train_char_gpt(path, **options())
    split, _, chunked_metrics = train_char_gpt(
        path, microbatch_size=micro_size, **options(),
    )
    assert chunked_metrics["microbatches_per_step"] == (5 + micro_size - 1) // micro_size
    assert chunked_metrics["effective_tokens_per_step"] == 15
    for key in ("train_loss_last", "validation_before", "validation_after"):
        assert chunked_metrics[key] == pytest.approx(full_metrics[key], abs=3e-5)
    for a, b in zip(full.parameters(), split.parameters()):
        torch.testing.assert_close(a, b, atol=3e-5, rtol=3e-5)


def test_resume_rejects_changed_microbatch_policy(tmp_path):
    path = corpus(tmp_path)
    checkpoint = tmp_path / "state.pt"
    args = options()
    args["steps"] = 1
    train_char_gpt(
        path, training_state_path=checkpoint,
        microbatch_size=2, **args,
    )
    args["steps"] = 2
    with pytest.raises(ValueError, match="controls"):
        train_char_gpt(path, resume_from=checkpoint, **args)
    resumed, _, metrics = train_char_gpt(
        path, resume_from=checkpoint, microbatch_size=2, **args,
    )
    assert next(resumed.parameters()).numel() > 0
    assert metrics["microbatches_per_step"] == 3


def test_rejects_incompatible_or_oversized_microbatches(tmp_path):
    path = corpus(tmp_path)
    with pytest.raises(ValueError, match="microbatch_size"):
        train_char_gpt(path, microbatch_size=6, **options())
    with pytest.raises(ValueError, match="cannot be combined"):
        train_char_gpt(
            path, microbatch_size=2, gradient_accumulation_steps=2,
            **options(),
        )


def test_cli_records_the_logical_microbatch_split(tmp_path):
    path = corpus(tmp_path)
    report = tmp_path / "history.json"
    argv = [
        "--corpus", str(path), "--steps", "1",
        "--block-size", "3", "--n-embd", "8", "--n-head", "2",
        "--n-layer", "1", "--batch-size", "5", "--microbatch-size", "2",
        "--eval-every", "1", "--eval-batches", "1",
        "--output", str(tmp_path / "model.pt"),
        "--best-output", str(tmp_path / "best.pt"),
        "--history-file", str(report),
    ]
    assert train_main(argv) == 0
    content = json.loads(report.read_text(encoding="utf-8"))
    assert content["training_controls"]["microbatch_size"] == 2
    assert content["training_controls"]["gradient_accumulation_steps"] == 1
    assert content["metrics"]["effective_tokens_per_step"] == 15
    assert content["metrics"]["microbatches_per_step"] == 3


def test_cli_rejects_conflicting_accumulation_modes():
    with pytest.raises(SystemExit) as error:
        train_main([
            "--batch-size", "5", "--microbatch-size", "2",
            "--gradient-accumulation-steps", "2",
        ])
    assert error.value.code == 2
