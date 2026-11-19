"""Fixed starts let us isolate data gathering from random sampling."""

import pytest
import torch

from rebuild_gpt.data import gather_next_token_pairs, sample_batch


def test_fixed_starts_match_existing_random_sampler_bitwise():
    tokens = torch.arange(80, dtype=torch.long)
    block_size = 5
    first_rng = torch.Generator().manual_seed(152)
    second_rng = torch.Generator().manual_seed(152)
    batch = sample_batch(
        tokens, batch_size=7, block_size=block_size, generator=first_rng,
    )
    starts = torch.randint(
        0, len(tokens) - block_size,
        (7,), generator=second_rng,
    )
    fixed = gather_next_token_pairs(tokens, starts, block_size=block_size)
    assert all(torch.equal(a, b) for a, b in zip(batch, fixed))
    assert torch.equal(first_rng.get_state(), second_rng.get_state())


def test_manual_contexts_and_targets_align():
    tokens = torch.arange(12)
    x, y = gather_next_token_pairs(
        tokens, torch.tensor([1, 4]), block_size=3,
    )
    assert x.tolist() == [[1, 2, 3], [4, 5, 6]]
    assert y.tolist() == [[2, 3, 4], [5, 6, 7]]


@pytest.mark.parametrize("starts", [
    torch.tensor([], dtype=torch.long),
    torch.tensor([1], dtype=torch.int32),
    torch.tensor([-1]),
    torch.tensor([17]),
])
def test_invalid_starts_are_rejected(starts):
    with pytest.raises(ValueError, match="starts"):
        gather_next_token_pairs(torch.arange(20), starts, block_size=3)


def test_invalid_block_length_is_rejected():
    with pytest.raises(ValueError, match="block_size"):
        gather_next_token_pairs(torch.arange(8), torch.tensor([0]), block_size=0)
