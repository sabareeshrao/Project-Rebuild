"""Preparing sampled evaluation batches must match the legacy sequence."""

import pytest
import torch

from rebuild_gpt.data import sample_batch
from rebuild_gpt.eval_cache import prepare_evaluation_batches


def test_preparation_matches_seeded_sampling_and_does_not_touch_global_rng():
    tokens = torch.arange(101, dtype=torch.long)
    before = torch.get_rng_state().clone()
    prepared = prepare_evaluation_batches(
        tokens, batch_size=3, block_size=5, eval_batches=4, seed=161,
    )
    assert torch.equal(before, torch.get_rng_state())
    original_rng = torch.Generator(device="cpu").manual_seed(161)
    assert len(prepared.batches) == 4
    for cached_x, cached_y in prepared.batches:
        expected_x, expected_y = sample_batch(
            tokens, batch_size=3, block_size=5, generator=original_rng,
        )
        assert torch.equal(cached_x, expected_x)
        assert torch.equal(cached_y, expected_y)
        assert cached_x.device.type == cached_y.device.type == "cpu"
    assert prepared.storage_bytes == 4 * 2 * 3 * 5 * 8


def test_prepared_batches_are_reusable_without_advancing_rng():
    tokens = torch.arange(72, dtype=torch.long)
    batches = prepare_evaluation_batches(
        tokens, batch_size=2, block_size=4, eval_batches=3, seed=9,
    )
    first = tuple((x.clone(), y.clone()) for x, y in batches.batches)
    state = torch.get_rng_state().clone()
    for (a, b), (x, y) in zip(first, batches.batches):
        assert torch.equal(a, x) and torch.equal(b, y)
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize("argument,bad", [
    ("batch_size", 0), ("batch_size", True),
    ("block_size", -1), ("eval_batches", 0),
    ("eval_batches", 1.5), ("seed", -1), ("seed", False),
])
def test_invalid_configuration_is_rejected(argument, bad):
    options = dict(batch_size=2, block_size=3, eval_batches=2, seed=1)
    options[argument] = bad
    with pytest.raises(ValueError, match=argument):
        prepare_evaluation_batches(torch.arange(30), **options)


def test_short_corpus_is_rejected():
    with pytest.raises(ValueError, match="block_size"):
        prepare_evaluation_batches(
            torch.arange(3), batch_size=2, block_size=3,
        )
