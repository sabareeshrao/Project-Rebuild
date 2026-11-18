import pytest
import torch

from rebuild_gpt.schedule import cosine_warmup_lr
from rebuild_gpt.text_training import train_char_gpt


def corpus(tmp_path):
    path = tmp_path / "ab.txt"
    path.write_text("ab" * 80, encoding="utf-8")
    return path


def test_schedule_and_clipping_are_reproducible(tmp_path):
    options = dict(steps=6, block_size=3, n_embd=8, n_head=2, n_layer=1,
                   learning_rate=0.03, warmup_steps=2, lr_decay_steps=5,
                   min_lr=0.003, max_grad_norm=0.5, seed=11)
    a, _, a_metrics = train_char_gpt(corpus(tmp_path), **options)
    b, _, b_metrics = train_char_gpt(corpus(tmp_path), **options)
    assert a_metrics == b_metrics
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    assert a_metrics["learning_rate_last"] == pytest.approx(
        cosine_warmup_lr(5, max_lr=0.03, min_lr=0.003,
                         warmup_steps=2, decay_steps=5)
    )


def test_scheduler_actually_changes_training_updates(tmp_path):
    path = corpus(tmp_path)
    fixed, _, _ = train_char_gpt(path, steps=5, block_size=3, n_embd=8,
                                 n_head=2, n_layer=1, seed=11)
    scheduled, _, _ = train_char_gpt(
        path, steps=5, block_size=3, n_embd=8, n_head=2, n_layer=1, seed=11,
        warmup_steps=2, lr_decay_steps=4, min_lr=0.001,
    )
    assert any(not torch.equal(a, b) for a, b in zip(fixed.parameters(), scheduled.parameters()))


def test_default_training_behavior_is_unchanged_by_optional_controls(tmp_path):
    path = corpus(tmp_path)
    options = dict(steps=4, block_size=3, n_embd=8, n_head=2, n_layer=1, seed=4)
    a, _, am = train_char_gpt(path, **options)
    b, _, bm = train_char_gpt(path, **options, max_grad_norm=None, lr_decay_steps=None)
    assert am == bm
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))


@pytest.mark.parametrize("kwargs,expected", [
    ({"max_grad_norm": 0}, "max_grad_norm"),
    ({"max_grad_norm": float("inf")}, "max_grad_norm"),
    ({"warmup_steps": 2}, "lr_decay_steps"),
    ({"min_lr": 0.001}, "lr_decay_steps"),
    ({"warmup_steps": 5, "lr_decay_steps": 5}, "decay_steps"),
    ({"warmup_steps": 2, "lr_decay_steps": 5, "min_lr": 3.0}, "min_lr"),
])
def test_invalid_training_controls_fail_early(tmp_path, kwargs, expected):
    with pytest.raises(ValueError, match=expected):
        train_char_gpt(corpus(tmp_path), steps=1, **kwargs)
