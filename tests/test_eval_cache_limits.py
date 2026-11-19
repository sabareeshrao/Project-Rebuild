"""Memory guards must reject plans before sampling either split."""

import pytest
import torch

from rebuild_gpt.data import sample_batch
from rebuild_gpt.eval_cache import (
    prepare_evaluation_batches, prepare_train_val_evaluation_batches,
)
from rebuild_gpt.eval_cache_budget import estimate_evaluation_cache_budget


def tokens():
    return torch.arange(80, dtype=torch.long) % 5


def test_single_split_limit_exactly_equal_to_estimate_is_accepted():
    estimate = estimate_evaluation_cache_budget(
        batch_size=3, block_size=4, eval_batches=2, split_count=1,
    )
    plan = prepare_evaluation_batches(
        tokens(), batch_size=3, block_size=4,
        eval_batches=2, seed=168, max_bytes=estimate.total_bytes,
    )
    assert plan.storage_bytes == estimate.total_bytes


def test_single_split_over_limit_rejects_before_sampling(monkeypatch):
    import rebuild_gpt.eval_cache as cache

    def forbidden(*args, **kwargs):
        raise AssertionError("Sampling must not start after budget rejection")

    monkeypatch.setattr(cache, "sample_batch", forbidden)
    with pytest.raises(ValueError, match="exceeding max_bytes"):
        prepare_evaluation_batches(
            tokens(), batch_size=3, block_size=4,
            eval_batches=2, seed=168, max_bytes=1,
        )


def test_combined_limit_checks_two_splits_before_any_samples(monkeypatch):
    import rebuild_gpt.eval_cache as cache

    calls = []
    original = cache.sample_batch

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(cache, "sample_batch", spy)
    estimate = estimate_evaluation_cache_budget(
        batch_size=2, block_size=4, eval_batches=3,
    )
    # A limit sufficient for either single split is not enough for two.
    with pytest.raises(ValueError, match="exceeding max_bytes"):
        prepare_train_val_evaluation_batches(
            tokens(), tokens(), batch_size=2,
            block_size=4, eval_batches=3, seed=168,
            max_bytes=estimate.bytes_per_split,
        )
    assert calls == []
    train, val = prepare_train_val_evaluation_batches(
        tokens(), tokens(), batch_size=2, block_size=4,
        eval_batches=3, seed=168, max_bytes=estimate.total_bytes,
    )
    assert len(calls) == 6
    assert train.seed == 168 and val.seed == 169
    assert train.storage_bytes + val.storage_bytes == estimate.total_bytes


def test_combined_preparation_matches_reference_independent_seeds():
    a, b = prepare_train_val_evaluation_batches(
        tokens(), tokens(), batch_size=3, block_size=4,
        eval_batches=2, seed=7,
    )
    for plan, seed in ((a, 7), (b, 8)):
        rng = torch.Generator(device="cpu").manual_seed(seed)
        for x, y in plan.batches:
            ex, ey = sample_batch(
                tokens(), batch_size=3, block_size=4, generator=rng,
            )
            assert torch.equal(x, ex) and torch.equal(y, ey)


@pytest.mark.parametrize("invalid", [0, -1, True, 0.5, "256"])
def test_invalid_byte_limits_are_rejected(invalid):
    with pytest.raises(ValueError, match="max_bytes"):
        prepare_train_val_evaluation_batches(
            tokens(), tokens(), batch_size=2, block_size=4,
            max_bytes=invalid,
        )
