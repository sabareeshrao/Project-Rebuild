import pytest
import torch

from rebuild_gpt.text_training import train_char_gpt


def test_small_repeating_corpus_can_train_and_reduces_validation_loss(tmp_path):
    path = tmp_path / "repeat.txt"
    path.write_text("ab" * 80, encoding="utf-8")
    model, tok, metrics = train_char_gpt(
        path, steps=35, block_size=4, n_embd=8, n_head=2, n_layer=1,
        batch_size=8, learning_rate=0.03, seed=4,
    )
    assert tok.vocab == ("a", "b")
    assert model.config.vocab_size == tok.vocab_size
    assert model.training is False
    assert metrics["validation_after"] < metrics["validation_before"]
    assert torch.isfinite(torch.tensor(list(metrics.values()))).all()


def test_text_training_is_reproducible_with_fixed_seed(tmp_path):
    path = tmp_path / "repeat.txt"
    path.write_text("ab" * 50, encoding="utf-8")
    options = dict(steps=4, block_size=3, n_embd=8, n_head=2, n_layer=1, seed=8)
    first, tok_a, result_a = train_char_gpt(path, **options)
    second, tok_b, result_b = train_char_gpt(path, **options)
    assert tok_a.stoi == tok_b.stoi
    assert result_a == result_b
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))


@pytest.mark.parametrize("kwargs,error", [
    ({"steps": 0}, "steps"),
    ({"batch_size": 0}, "batch_size"),
    ({"learning_rate": 0}, "learning_rate"),
    ({"seed": -1}, "seed"),
])
def test_training_rejects_invalid_options(tmp_path, kwargs, error):
    path = tmp_path / "repeat.txt"
    path.write_text("ab" * 80, encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        train_char_gpt(path, **kwargs)


def test_training_rejects_corpus_shorter_than_context(tmp_path):
    path = tmp_path / "short.txt"
    path.write_text("abbaab", encoding="utf-8")
    with pytest.raises(ValueError, match="too short"):
        train_char_gpt(path, block_size=4, steps=1)
