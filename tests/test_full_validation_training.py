import pytest
import torch

from rebuild_gpt.corpus_evaluation import full_corpus_loss
from rebuild_gpt.data import load_corpus, split_train_val
from rebuild_gpt.text_checkpoint import load_text_checkpoint
from rebuild_gpt.text_training import train_char_gpt


def corpus(tmp_path):
    path = tmp_path / "text.txt"
    path.write_text("ab" * 110, encoding="utf-8")
    return path


def test_full_validation_uses_entire_split_for_best_checkpoint(tmp_path):
    path = corpus(tmp_path)
    best_file = tmp_path / "best.pt"
    history = []
    model, tokenizer, metrics = train_char_gpt(
        path, steps=5, block_size=4, n_embd=8, n_head=2, n_layer=1,
        batch_size=4, eval_interval=2, validation_mode="full",
        restore_best=True, best_checkpoint_path=best_file, history=history,
    )
    _, val = split_train_val(load_corpus(path)[0], val_fraction=0.2)
    assert [r["step"] for r in history] == [0, 2, 4, 5]
    assert metrics["best_validation_loss"] == min(row["validation_loss"] for row in history)
    assert metrics["validation_after"] == pytest.approx(
        full_corpus_loss(model, val, block_size=4), abs=1e-8
    )
    assert metrics["validation_after"] == pytest.approx(metrics["best_validation_loss"], abs=1e-8)
    loaded, restored_tok = load_text_checkpoint(best_file)
    assert restored_tok.stoi == tokenizer.stoi
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), loaded.parameters()))


def test_full_validation_is_reproducible_and_does_not_change_updates(tmp_path):
    path = corpus(tmp_path)
    params = dict(steps=4, block_size=3, n_embd=8, n_head=2, n_layer=1, seed=5)
    original, _, _ = train_char_gpt(path, **params)
    full, _, _ = train_char_gpt(path, **params, validation_mode="full", eval_interval=2)
    assert all(torch.equal(a, b) for a, b in zip(original.parameters(), full.parameters()))


def test_invalid_validation_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="validation_mode"):
        train_char_gpt(corpus(tmp_path), steps=1, validation_mode="unknown")
