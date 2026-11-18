import pytest
import torch

from rebuild_gpt.text_training import train_char_gpt


def corpus(tmp_path):
    path = tmp_path / "repeat.txt"
    path.write_text("ab" * 110, encoding="utf-8")
    return path


def test_accumulated_training_uses_microbatches_and_is_reproducible(tmp_path):
    options = dict(steps=5, block_size=4, batch_size=3, n_embd=8, n_head=2,
                   n_layer=1, seed=5, gradient_accumulation_steps=3,
                   grouped_adamw=True, max_grad_norm=1.0)
    a, _, metrics_a = train_char_gpt(corpus(tmp_path), **options)
    b, _, metrics_b = train_char_gpt(corpus(tmp_path), **options)
    assert metrics_a == metrics_b
    assert metrics_a["effective_tokens_per_step"] == 36
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))


def test_one_microbatch_retains_legacy_result_exactly(tmp_path):
    options = dict(steps=4, block_size=3, n_embd=8, n_head=2, n_layer=1, seed=7)
    before, _, old_metrics = train_char_gpt(corpus(tmp_path), **options)
    one, _, one_metrics = train_char_gpt(corpus(tmp_path), **options, gradient_accumulation_steps=1)
    assert old_metrics == one_metrics
    assert all(torch.equal(x, y) for x, y in zip(before.parameters(), one.parameters()))


def test_accumulation_means_more_data_per_optimizer_step(tmp_path):
    options = dict(steps=5, block_size=3, n_embd=8, n_head=2, n_layer=1, seed=10)
    first, _, _ = train_char_gpt(corpus(tmp_path), **options)
    three, _, _ = train_char_gpt(corpus(tmp_path), **options, gradient_accumulation_steps=3)
    assert any(not torch.equal(x, y) for x, y in zip(first.parameters(), three.parameters()))


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_invalid_gradient_accumulation_steps(tmp_path, bad):
    with pytest.raises(ValueError, match="gradient_accumulation_steps"):
        train_char_gpt(corpus(tmp_path), steps=1, gradient_accumulation_steps=bad)
