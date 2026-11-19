"""Optional host cache limits protect resumable character-GPT evaluation."""

import pytest
import torch

from rebuild_gpt.text_training import train_char_gpt


def corpus(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("abc" * 125, encoding="utf-8")
    return path


def options():
    return dict(
        steps=3, batch_size=4, block_size=3,
        n_embd=8, n_head=2, n_layer=1,
        eval_interval=1, eval_batches=2, seed=169,
        attention_impl="sdpa",
        eval_batch_backend="cached",
    )


def test_sufficient_cache_budget_matches_uncapped_evaluation(tmp_path):
    text = corpus(tmp_path)
    # 2 splits × 2 (inputs + targets) × 4 × 3 × 2 × 8 = 384 bytes.
    bound_mib = 384 / (1024 * 1024)
    reference_history, bounded_history = [], []
    original, _, a = train_char_gpt(
        text, history=reference_history, **options(),
    )
    guarded, _, b = train_char_gpt(
        text, history=bounded_history, eval_cache_max_mib=bound_mib,
        **options(),
    )
    assert reference_history == bounded_history
    assert a == b
    assert all(torch.equal(x, y)
               for x, y in zip(original.parameters(), guarded.parameters()))


def test_oversized_cache_is_rejected_before_batch_sampling(tmp_path, monkeypatch):
    import rebuild_gpt.eval_cache as cache

    def forbidden(*args, **kwargs):
        raise AssertionError("Cache budget should fail before sampling")

    monkeypatch.setattr(cache, "sample_batch", forbidden)
    with pytest.raises(ValueError, match="exceeding max_bytes"):
        train_char_gpt(
            corpus(tmp_path), eval_cache_max_mib=0.0001, **options(),
        )


def test_resume_requires_same_byte_budget(tmp_path):
    text = corpus(tmp_path)
    state = tmp_path / "saved.pt"
    args = options()
    args["steps"] = 2
    budget = 384 / (1024 * 1024)
    train_char_gpt(
        text, eval_cache_max_mib=budget,
        training_state_path=state, **args,
    )
    args["steps"] = 4
    with pytest.raises(ValueError, match="controls"):
        train_char_gpt(text, resume_from=state, **args)
    with pytest.raises(ValueError, match="controls"):
        train_char_gpt(
            text, eval_cache_max_mib=budget * 2,
            resume_from=state, **args,
        )
    resumed, _, score = train_char_gpt(
        text, eval_cache_max_mib=budget,
        resume_from=state, **args,
    )
    full, _, reference = train_char_gpt(
        text, eval_cache_max_mib=budget, **args,
    )
    assert score == reference
    assert all(torch.equal(a, b) for a, b in zip(resumed.parameters(), full.parameters()))


@pytest.mark.parametrize("cap", [0, -1, True, float("inf"), float("nan"), "1"])
def test_invalid_cache_limit_rejected(tmp_path, cap):
    with pytest.raises(ValueError, match="eval_cache_max_mib"):
        train_char_gpt(corpus(tmp_path), eval_cache_max_mib=cap, **options())


def test_budget_cannot_be_ignored_by_regenerate_backend(tmp_path):
    args = options()
    args["eval_batch_backend"] = "regenerate"
    with pytest.raises(ValueError, match="requires cached"):
        train_char_gpt(corpus(tmp_path), eval_cache_max_mib=1.0, **args)
