"""Index plans enforce a combined CPU memory cap before drawing starts."""

import pytest
import torch

from rebuild_gpt.eval_index_plan import (
    estimate_index_plan_bytes, prepare_train_val_index_plans,
)


def test_estimator_counts_start_vectors_only():
    assert estimate_index_plan_bytes(
        batch_size=3, eval_batches=4,
    ) == 2 * 3 * 4 * 8
    assert estimate_index_plan_bytes(
        batch_size=3, eval_batches=4, split_count=1,
    ) == 3 * 4 * 8


def test_exact_budget_is_accepted_for_both_splits():
    tokens = torch.arange(100, dtype=torch.long)
    limit = estimate_index_plan_bytes(batch_size=3, eval_batches=2)
    train, val = prepare_train_val_index_plans(
        tokens, tokens, batch_size=3, block_size=6,
        eval_batches=2, seed=174, max_bytes=limit,
    )
    assert train.index_storage_bytes + val.index_storage_bytes == limit
    assert train.seed == 174 and val.seed == 175


def test_combined_limit_rejects_before_any_random_draw(monkeypatch):
    original = torch.randint
    calls = []
    def spy(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, "randint", spy)
    tokens = torch.arange(80)
    individual = estimate_index_plan_bytes(
        batch_size=2, eval_batches=3, split_count=1,
    )
    with pytest.raises(ValueError, match="exceeding max_bytes"):
        prepare_train_val_index_plans(
            tokens, tokens, batch_size=2, block_size=4,
            eval_batches=3, max_bytes=individual,
        )
    assert calls == []


def test_invalid_validation_corpus_rejected_before_random_draw(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("index sampling must not start")
    monkeypatch.setattr(torch, "randint", forbidden)
    with pytest.raises(ValueError, match="block_size"):
        prepare_train_val_index_plans(
            torch.arange(70), torch.arange(3),
            batch_size=2, block_size=4,
        )


@pytest.mark.parametrize("invalid", [0, -1, True, 1.5, "24"])
def test_invalid_limit_is_rejected(invalid):
    with pytest.raises(ValueError, match="max_bytes"):
        prepare_train_val_index_plans(
            torch.arange(80), torch.arange(80),
            batch_size=2, block_size=4, max_bytes=invalid,
        )


@pytest.mark.parametrize("name,bad", [
    ("batch_size", 0), ("eval_batches", -1),
    ("split_count", True), ("split_count", 2.5),
])
def test_bad_estimator_inputs_are_rejected(name, bad):
    kwargs = dict(batch_size=3, eval_batches=4, split_count=2)
    kwargs[name] = bad
    with pytest.raises(ValueError, match=name):
        estimate_index_plan_bytes(**kwargs)
