"""The cache budget is conservative, deterministic and GPU independent."""

import pytest
import torch

from rebuild_gpt.eval_cache_budget import (
    enforce_evaluation_cache_budget, estimate_evaluation_cache_budget,
)


def test_budget_counts_both_input_and_target_tensors_across_splits():
    plan = estimate_evaluation_cache_budget(
        batch_size=3, block_size=7, eval_batches=4, split_count=2,
    )
    assert plan.bytes_per_split == 2 * 3 * 7 * 4 * 8
    assert plan.total_bytes == 2 * plan.bytes_per_split
    assert plan.total_mib == pytest.approx(plan.total_bytes / 1048576)
    assert torch.tensor([], dtype=torch.long).element_size() == 8


def test_budget_is_proportional_to_requested_evaluations():
    a = estimate_evaluation_cache_budget(
        batch_size=2, block_size=5, eval_batches=2,
    )
    b = estimate_evaluation_cache_budget(
        batch_size=2, block_size=5, eval_batches=6,
    )
    assert b.total_bytes == a.total_bytes * 3


def test_limit_is_inclusive_and_can_be_disabled():
    plan = estimate_evaluation_cache_budget(
        batch_size=2, block_size=4, eval_batches=2,
    )
    enforce_evaluation_cache_budget(plan.total_bytes, plan.total_bytes)
    enforce_evaluation_cache_budget(plan.total_bytes, None)
    with pytest.raises(ValueError, match="exceeding max_bytes"):
        enforce_evaluation_cache_budget(plan.total_bytes, plan.total_bytes - 1)


@pytest.mark.parametrize("field,invalid", [
    ("batch_size", 0), ("batch_size", True),
    ("block_size", -1), ("eval_batches", 2.5),
    ("split_count", 0), ("split_count", "2"),
])
def test_estimator_rejects_invalid_shapes(field, invalid):
    options = dict(batch_size=2, block_size=4, eval_batches=2, split_count=2)
    options[field] = invalid
    with pytest.raises(ValueError, match=field):
        estimate_evaluation_cache_budget(**options)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "128"])
def test_limit_rejects_nonpositive_and_noninteger_values(bad):
    with pytest.raises(ValueError, match="max_bytes"):
        enforce_evaluation_cache_budget(64, bad)


def test_budget_describes_logical_bytes_not_process_rss():
    # Adjacent next-token targets overlap with inputs in their base
    # [B,T+1] sampled window, so this budget deliberately overcounts.
    batch = torch.arange(12).reshape(2, 6)
    x, y = batch[:, :-1], batch[:, 1:]
    assert x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()
    assert x.numel() * 8 + y.numel() * 8 >= batch.untyped_storage().nbytes()
