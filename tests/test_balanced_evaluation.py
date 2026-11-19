"""Region-balanced starts are deterministic and explicit about capacity."""

import pytest
import torch

from rebuild_gpt.balanced_evaluation import (
    prepare_balanced_evaluation_index_plan, start_region_bounds,
)
from rebuild_gpt.window_overlap import analyze_index_plan_overlap


def test_start_region_bounds_partition_all_valid_start_indices():
    bounds = start_region_bounds(
        corpus_token_count=25, block_size=4, region_count=4,
    )
    assert bounds == ((0, 5), (5, 10), (10, 15), (15, 21))
    assert [value for start, stop in bounds for value in range(start, stop)] == list(
        range(21)
    )


def test_balanced_index_plan_is_repeatable_with_private_generator():
    tokens = torch.arange(170)
    rng = torch.get_rng_state().clone()
    opts = dict(batch_size=3, block_size=6, eval_batches=4, seed=214, region_count=5)
    first = prepare_balanced_evaluation_index_plan(tokens, **opts)
    second = prepare_balanced_evaluation_index_plan(tokens, **opts)
    assert first.tokens is tokens
    assert first.batch_size == 3 and first.block_size == 6
    assert first.index_storage_bytes == 12 * 8
    assert all(torch.equal(a, b) for a, b in zip(first.starts, second.starts))
    assert all(
        torch.equal(x[:, 1:], y[:, :-1]) for x, y in first.batches()
    )
    assert torch.equal(rng, torch.get_rng_state())


def test_every_start_region_is_sampled_with_nearly_equal_counts():
    plan = prepare_balanced_evaluation_index_plan(
        torch.arange(1000), batch_size=3, block_size=8,
        eval_batches=4, seed=12, region_count=5,
    )
    starts = torch.cat(plan.starts).tolist()
    bounds = start_region_bounds(
        corpus_token_count=1000, block_size=8, region_count=5,
    )
    counts = [sum(low <= value < high for value in starts) for low, high in bounds]
    assert counts == [3, 3, 2, 2, 2]
    assert sum(counts) == 12
    assert analyze_index_plan_overlap(plan).window_count == 12


def test_exact_one_window_per_region_is_allowed():
    plan = prepare_balanced_evaluation_index_plan(
        torch.arange(15), batch_size=2, block_size=3,
        eval_batches=2, region_count=4, seed=4,
    )
    starts = torch.cat(plan.starts).tolist()
    bounds = start_region_bounds(
        corpus_token_count=15, block_size=3, region_count=4,
    )
    assert all(sum(lo <= x < hi for x in starts) == 1 for lo, hi in bounds)


@pytest.mark.parametrize("name,bad", [
    ("batch_size", 0), ("block_size", True),
    ("eval_batches", -1), ("region_count", 1.5),
    ("seed", -1), ("seed", False),
])
def test_invalid_parameters_rejected(name, bad):
    args = dict(batch_size=2, block_size=3, eval_batches=3,
                region_count=4, seed=214)
    args[name] = bad
    with pytest.raises(ValueError, match=name):
        prepare_balanced_evaluation_index_plan(torch.arange(70), **args)


def test_short_corpus_or_too_many_regions_is_explicit_error():
    with pytest.raises(ValueError, match="corpus_token_count"):
        prepare_balanced_evaluation_index_plan(
            torch.arange(3), batch_size=2, block_size=3,
        )
    with pytest.raises(ValueError, match="region_count"):
        prepare_balanced_evaluation_index_plan(
            torch.arange(10), batch_size=3, block_size=8, region_count=3,
        )
    with pytest.raises(ValueError, match="at least one window"):
        prepare_balanced_evaluation_index_plan(
            torch.arange(30), batch_size=1, block_size=4,
            eval_batches=2, region_count=3,
        )


def test_non_int64_tokens_are_rejected():
    with pytest.raises(ValueError, match="torch.long"):
        prepare_balanced_evaluation_index_plan(
            torch.arange(50, dtype=torch.int32),
            batch_size=2, block_size=3,
        )
