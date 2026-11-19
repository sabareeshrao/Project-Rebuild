"""Start-only evaluation plans retain RNG parity with full-batch sampling."""

import pytest
import torch

from rebuild_gpt.data import sample_batch
from rebuild_gpt.eval_index_plan import prepare_evaluation_index_plan


@pytest.mark.parametrize("batch_size,context,eval_batches", [
    (1, 1, 3), (3, 4, 2), (5, 8, 4),
])
def test_index_plan_yields_reference_batches_without_global_rng_change(
    batch_size, context, eval_batches,
):
    tokens = torch.arange(101, dtype=torch.long) % 13
    before = torch.get_rng_state().clone()
    plan = prepare_evaluation_index_plan(
        tokens, batch_size=batch_size, block_size=context,
        eval_batches=eval_batches, seed=173,
    )
    assert torch.equal(before, torch.get_rng_state())
    assert plan.tokens is tokens
    assert plan.index_storage_bytes == eval_batches * batch_size * 8
    generator = torch.Generator(device="cpu").manual_seed(173)
    expected = [
        sample_batch(
            tokens, batch_size=batch_size, block_size=context,
            generator=generator,
        )
        for _ in range(eval_batches)
    ]
    for (actual_x, actual_y), (expected_x, expected_y) in zip(
        plan.batches(), expected,
    ):
        assert torch.equal(actual_x, expected_x)
        assert torch.equal(actual_y, expected_y)
    assert torch.equal(before, torch.get_rng_state())
    for first, second in zip(plan.batches(), plan.batches()):
        assert all(torch.equal(x, y) for x, y in zip(first, second))


def test_start_positions_are_smaller_than_materialized_token_pairs():
    plan = prepare_evaluation_index_plan(
        torch.arange(1000), batch_size=4, block_size=32, eval_batches=5,
    )
    assert plan.index_storage_bytes == 4 * 5 * 8
    assert plan.index_storage_bytes < 2 * 4 * 32 * 5 * 8


@pytest.mark.parametrize("name,value", [
    ("batch_size", 0), ("batch_size", True), ("block_size", 0),
    ("eval_batches", -1), ("seed", -1), ("seed", 3.2),
])
def test_rejects_invalid_counts(name, value):
    kwargs = dict(batch_size=2, block_size=3, eval_batches=2, seed=17)
    kwargs[name] = value
    with pytest.raises(ValueError, match=name):
        prepare_evaluation_index_plan(torch.arange(40), **kwargs)


def test_rejects_short_context_and_invalid_dtype():
    with pytest.raises(ValueError, match="block_size"):
        prepare_evaluation_index_plan(
            torch.arange(4), batch_size=2, block_size=4,
        )
    with pytest.raises(ValueError, match="torch.long"):
        prepare_evaluation_index_plan(
            torch.arange(12, dtype=torch.int32), batch_size=2, block_size=3,
        )
