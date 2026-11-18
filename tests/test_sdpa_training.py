"""Train and reload the same GPT architecture with PyTorch SDPA."""

import pytest
import torch

from rebuild_gpt.text_checkpoint import load_text_checkpoint
from rebuild_gpt.text_training import train_char_gpt
from rebuild_gpt.train_cli import main as train_main


def corpus(tmp_path):
    path = tmp_path / "text.txt"
    path.write_text("ab" * 110, encoding="utf-8")
    return path


def test_sdpa_training_runs_with_finite_loss(tmp_path):
    model, tokenizer, result = train_char_gpt(
        corpus(tmp_path), steps=2, block_size=3,
        n_embd=8, n_head=2, n_layer=1,
        batch_size=3, eval_interval=1, eval_batches=1,
        attention_impl="sdpa", seed=9,
    )
    assert model.config.attention_impl == "sdpa"
    assert tokenizer.vocab_size == 2
    assert torch.isfinite(torch.tensor(list(result.values()))).all()


def test_attention_backend_is_checkpointed_and_verified_on_resume(tmp_path):
    path = corpus(tmp_path)
    state = tmp_path / "resume.pt"
    opts = dict(steps=2, block_size=3, n_embd=8, n_head=2,
                n_layer=1, batch_size=3)
    train_char_gpt(path, attention_impl="sdpa",
                   training_state_path=state, **opts)
    with pytest.raises(ValueError, match="config"):
        train_char_gpt(path, resume_from=state, steps=4, block_size=3,
                       n_embd=8, n_head=2, n_layer=1, batch_size=3,
                       attention_impl="fused")


def test_cli_sdpa_checkpoint_loads_with_same_attention_impl(tmp_path):
    path = corpus(tmp_path)
    out = tmp_path / "final.pt"
    args = [
        "--corpus", str(path), "--steps", "2", "--block-size", "3",
        "--batch-size", "3", "--n-embd", "8", "--n-head", "2",
        "--n-layer", "1", "--attention-impl", "sdpa",
        "--eval-every", "1", "--eval-batches", "1",
        "--output", str(out), "--best-output", str(tmp_path / "best.pt"),
        "--history-file", str(tmp_path / "metrics.json"),
    ]
    assert train_main(args) == 0
    reloaded, _ = load_text_checkpoint(out)
    assert reloaded.config.attention_impl == "sdpa"


def test_invalid_text_training_attention_impl_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="attention_impl"):
        train_char_gpt(corpus(tmp_path), steps=1, attention_impl="educational")
