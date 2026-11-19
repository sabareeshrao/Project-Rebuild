"""Caching evaluation batches must preserve reproducibility and model updates."""

import pytest
import torch

from rebuild_gpt.text_training import train_char_gpt


def corpus(tmp_path):
    path = tmp_path / "text.txt"
    path.write_text("abc" * 110, encoding="utf-8")
    return path


def options():
    return dict(
        steps=3, batch_size=4, block_size=3, n_embd=8, n_head=2,
        n_layer=1, eval_interval=1, eval_batches=2, seed=163,
        attention_impl="sdpa",
    )


@pytest.mark.parametrize("extra", [
    {},
    {"sampling_backend": "cached"},
    {"microbatch_size": 2},
    {"gradient_accumulation_steps": 2},
])
def test_cached_evaluation_preserves_full_training_trajectory(tmp_path, extra):
    text = corpus(tmp_path)
    old_hist, cache_hist = [], []
    a, _, a_metrics = train_char_gpt(
        text, eval_batch_backend="regenerate", history=old_hist,
        **options(), **extra,
    )
    b, _, b_metrics = train_char_gpt(
        text, eval_batch_backend="cached", history=cache_hist,
        **options(), **extra,
    )
    assert a_metrics == b_metrics
    assert old_hist == cache_hist
    for w1, w2 in zip(a.state_dict().values(), b.state_dict().values()):
        assert torch.equal(w1, w2)


def test_eval_cache_is_prepared_once_per_split(tmp_path, monkeypatch):
    import rebuild_gpt.text_training as training

    called = []
    original = training.prepare_train_val_evaluation_batches

    def spy(train, validation, **kwargs):
        called.append((train.numel(), validation.numel(), kwargs["seed"]))
        return original(train, validation, **kwargs)

    monkeypatch.setattr(training, "prepare_train_val_evaluation_batches", spy)
    train_char_gpt(corpus(tmp_path), eval_batch_backend="cached", **options())
    assert len(called) == 1
    assert called[0][2] == 165


def test_resume_preserves_cached_evaluation_policy_and_weights(tmp_path):
    text = corpus(tmp_path)
    checkpoint = tmp_path / "resume.pt"
    first = options()
    first["steps"] = 2
    train_char_gpt(
        text, eval_batch_backend="cached",
        training_state_path=checkpoint, **first,
    )
    next_steps = options()
    next_steps["steps"] = 4
    with pytest.raises(ValueError, match="controls"):
        train_char_gpt(text, resume_from=checkpoint, **next_steps)
    resumed, _, scores = train_char_gpt(
        text, eval_batch_backend="cached",
        resume_from=checkpoint, **next_steps,
    )
    full, _, original_scores = train_char_gpt(
        text, eval_batch_backend="cached", **next_steps,
    )
    assert scores == original_scores
    for a, b in zip(resumed.parameters(), full.parameters()):
        assert torch.equal(a, b)


@pytest.mark.parametrize("changes,pattern", [
    ({"eval_batch_backend": "unknown"}, "eval_batch_backend"),
    ({"eval_batch_backend": "cached", "validation_mode": "full"}, "sampled"),
    ({"eval_batch_backend": "cached", "eval_interval": None}, "eval_interval"),
])
def test_incompatible_evaluation_controls_are_rejected(tmp_path, changes, pattern):
    settings = options()
    settings.update(changes)
    with pytest.raises(ValueError, match=pattern):
        train_char_gpt(corpus(tmp_path), **settings)
