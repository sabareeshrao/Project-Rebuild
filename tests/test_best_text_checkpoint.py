import pytest
import torch

from rebuild_gpt.evaluation import estimate_train_val_loss
from rebuild_gpt.text_checkpoint import load_text_checkpoint
from rebuild_gpt.text_training import train_char_gpt


def corpus(tmp_path):
    path = tmp_path / "repeat.txt"
    path.write_text("ab" * 100, encoding="utf-8")
    return path


def test_saved_best_checkpoint_matches_restored_best_model(tmp_path):
    path = tmp_path / "best.pt"
    history = []
    model, tokenizer, metrics = train_char_gpt(
        corpus(tmp_path), steps=12, block_size=4, n_embd=8,
        n_head=2, n_layer=1, batch_size=4, seed=5,
        eval_interval=4, eval_batches=2, history=history,
        best_checkpoint_path=path, restore_best=True,
    )
    saved, saved_tok = load_text_checkpoint(path)
    assert path.exists()
    assert saved_tok.vocab == tokenizer.vocab
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), saved.parameters()))
    minimum = min(entry["validation_loss"] for entry in history)
    best_record = next(entry for entry in history if entry["validation_loss"] == minimum)
    assert metrics["best_validation_loss"] == minimum
    assert metrics["best_step"] == best_record["step"]
    assert model.training is False


def test_best_model_can_be_saved_without_restoring_final_weights(tmp_path):
    path = tmp_path / "best.pt"
    _, _, metrics = train_char_gpt(
        corpus(tmp_path), steps=5, block_size=4, n_embd=8,
        n_head=2, n_layer=1, eval_interval=3,
        best_checkpoint_path=path, seed=4,
    )
    loaded, _ = load_text_checkpoint(path)
    assert loaded.config.block_size == 4
    assert 0 <= metrics["best_step"] <= 5


@pytest.mark.parametrize("kwargs", [
    {"restore_best": True},
    {"best_checkpoint_path": "unused.pt"},
])
def test_best_options_require_periodic_evaluation(tmp_path, kwargs):
    with pytest.raises(ValueError, match="requires eval_interval"):
        train_char_gpt(corpus(tmp_path), steps=1, **kwargs)


def test_best_metric_is_repeatable_with_same_seed(tmp_path):
    params = dict(steps=4, block_size=3, n_embd=8, n_head=2, n_layer=1,
                  eval_interval=2, restore_best=True, eval_batches=2, seed=11)
    a, _, x = train_char_gpt(corpus(tmp_path), **params)
    b, _, y = train_char_gpt(corpus(tmp_path), **params)
    assert x == y
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
