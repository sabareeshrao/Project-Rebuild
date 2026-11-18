import pytest
import torch

from rebuild_gpt.text_training import train_char_gpt


def corpus(tmp_path):
    file = tmp_path / "ab.txt"
    file.write_text("ab" * 110, encoding="utf-8")
    return file


def test_grouped_adamw_trains_and_is_deterministic(tmp_path):
    options = dict(steps=6, block_size=3, n_embd=8, n_head=2, n_layer=1,
                   seed=11, grouped_adamw=True, weight_decay=0.1,
                   adam_betas=(0.9, 0.95))
    a, _, am = train_char_gpt(corpus(tmp_path), **options)
    b, _, bm = train_char_gpt(corpus(tmp_path), **options)
    assert am == bm
    assert am["optimizer_group_count"] == 2
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


def test_opt_in_grouped_adamw_differs_from_legacy_optimizer(tmp_path):
    options = dict(steps=5, block_size=3, n_embd=8, n_head=2, n_layer=1, seed=8)
    legacy, _, metrics = train_char_gpt(corpus(tmp_path), **options)
    grouped, _, grouped_metrics = train_char_gpt(corpus(tmp_path), **options, grouped_adamw=True)
    assert "optimizer_group_count" not in metrics
    assert grouped_metrics["optimizer_group_count"] == 2
    assert any(not torch.equal(p, q) for p, q in zip(legacy.parameters(), grouped.parameters()))


def test_default_optimizer_unchanged(tmp_path):
    options = dict(steps=3, block_size=3, n_embd=8, n_head=2, n_layer=1, seed=4)
    first, _, m1 = train_char_gpt(corpus(tmp_path), **options)
    second, _, m2 = train_char_gpt(corpus(tmp_path), **options, grouped_adamw=False)
    assert m1 == m2
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))


@pytest.mark.parametrize("kwargs,match", [
    ({"grouped_adamw": 1}, "grouped_adamw"),
    ({"weight_decay": 0.2}, "requires grouped_adamw"),
    ({"adam_betas": (0.8, 0.9)}, "requires grouped_adamw"),
    ({"grouped_adamw": True, "weight_decay": -1}, "weight_decay"),
])
def test_grouped_optimizer_options_validate(tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        train_char_gpt(corpus(tmp_path), steps=1, **kwargs)
