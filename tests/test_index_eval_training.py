"""Index-only validation must preserve character-GPT learning and resumes."""

import pytest
import torch

from rebuild_gpt.text_training import train_char_gpt


def corpus(tmp_path):
    path = tmp_path / "story.txt"
    path.write_text("abc" * 130, encoding="utf-8")
    return path


def opts():
    return dict(
        steps=3, batch_size=3, block_size=4,
        n_embd=8, n_head=2, n_layer=1,
        eval_interval=1, eval_batches=2, seed=176,
        attention_impl="sdpa",
    )


@pytest.mark.parametrize("extra", [
    {}, {"sampling_backend": "cached"},
    {"microbatch_size": 2}, {"gradient_accumulation_steps": 2},
])
def test_index_plan_training_matches_regenerate_and_materialized(tmp_path, extra):
    path = corpus(tmp_path)
    baselines = []
    for backend in ("regenerate", "cached", "indices"):
        history = []
        model, _, scores = train_char_gpt(
            path, history=history, eval_batch_backend=backend,
            **opts(), **extra,
        )
        baselines.append((model, history, scores))
    first, hist, scores = baselines[0]
    for other, other_history, other_scores in baselines[1:]:
        assert hist == other_history
        assert scores == other_scores
        for a, b in zip(first.state_dict().values(), other.state_dict().values()):
            assert torch.equal(a, b)


def test_small_index_budget_accepts_what_materialized_cache_rejects(tmp_path):
    path = corpus(tmp_path)
    # Two splits × B=3 × K=2 × 8 bytes = 96 retained index bytes.
    byte_limit = 96 / 1048576
    model, _, scores = train_char_gpt(
        path, eval_batch_backend="indices",
        eval_cache_max_mib=byte_limit, **opts(),
    )
    assert scores["validation_after"] > 0
    assert next(model.parameters()).numel() > 0
    with pytest.raises(ValueError, match="exceeding max_bytes"):
        train_char_gpt(
            path, eval_batch_backend="cached",
            eval_cache_max_mib=byte_limit, **opts(),
        )


def test_resume_rejects_switching_index_cache_policy_and_cap(tmp_path):
    path = corpus(tmp_path)
    state = tmp_path / "train.pt"
    first = opts()
    first["steps"] = 2
    cap = 96 / 1048576
    train_char_gpt(
        path, eval_batch_backend="indices", eval_cache_max_mib=cap,
        training_state_path=state, **first,
    )
    later = opts()
    later["steps"] = 4
    with pytest.raises(ValueError, match="controls"):
        train_char_gpt(
            path, eval_batch_backend="cached",
            resume_from=state, **later,
        )
    with pytest.raises(ValueError, match="controls"):
        train_char_gpt(
            path, eval_batch_backend="indices",
            resume_from=state, **later,
        )
    resumed, _, resumed_scores = train_char_gpt(
        path, eval_batch_backend="indices", eval_cache_max_mib=cap,
        resume_from=state, **later,
    )
    full, _, full_scores = train_char_gpt(
        path, eval_batch_backend="indices", eval_cache_max_mib=cap, **later,
    )
    assert resumed_scores == full_scores
    assert all(torch.equal(a, b)
               for a, b in zip(resumed.parameters(), full.parameters()))


def test_index_backend_only_supports_sampled_validation(tmp_path):
    with pytest.raises(ValueError, match="sampled"):
        train_char_gpt(
            corpus(tmp_path), eval_batch_backend="indices",
            validation_mode="full", **opts(),
        )
