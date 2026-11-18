import pytest
import torch

from rebuild_gpt.text_training import train_char_gpt


def corpus(tmp_path):
    path = tmp_path / "repeat.txt"
    path.write_text("ab" * 90, encoding="utf-8")
    return path


def test_history_includes_initial_periodic_and_final_evaluations(tmp_path):
    history = [{"step": -1, "train_loss": 0, "validation_loss": 0}]
    model, tok, metrics = train_char_gpt(
        corpus(tmp_path), steps=7, block_size=4, n_embd=8, n_head=2,
        n_layer=1, batch_size=4, seed=7,
        eval_interval=3, eval_batches=2, history=history,
    )
    assert [entry["step"] for entry in history] == [0, 3, 6, 7]
    assert all(entry["train_loss"] > 0 and entry["validation_loss"] > 0 for entry in history)
    assert model.training is False
    assert "validation_after" in metrics


def test_tracking_does_not_change_model_weights_or_legacy_metrics(tmp_path):
    path = corpus(tmp_path)
    options = dict(steps=5, block_size=4, n_embd=8, n_head=2, n_layer=1, seed=19)
    original, _, before = train_char_gpt(path, **options)
    history = []
    tracked, _, after = train_char_gpt(
        path, **options, eval_interval=2, eval_batches=3, history=history
    )
    assert before == after
    assert all(torch.equal(a, b) for a, b in zip(original.parameters(), tracked.parameters()))
    assert [entry["step"] for entry in history] == [0, 2, 4, 5]


def test_history_is_reproducible_for_identical_seed(tmp_path):
    a, b = [], []
    kwargs = dict(steps=4, block_size=3, n_embd=8, n_head=2, n_layer=1, eval_interval=2)
    train_char_gpt(corpus(tmp_path), **kwargs, history=a)
    train_char_gpt(corpus(tmp_path), **kwargs, history=b)
    assert a == b


@pytest.mark.parametrize("kwargs,error", [
    ({"eval_interval": 0}, "eval_interval"),
    ({"eval_batches": 0}, "eval_batches"),
    ({"history": []}, "requires eval_interval"),
])
def test_invalid_tracking_settings_are_rejected(tmp_path, kwargs, error):
    with pytest.raises(ValueError, match=error):
        train_char_gpt(corpus(tmp_path), steps=1, **kwargs)
