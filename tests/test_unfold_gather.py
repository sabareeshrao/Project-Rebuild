"""Unfold-based window gathering is bit-identical to indexed gathering."""

import pytest
import torch

from rebuild_gpt.data import (
    gather_next_token_pairs, gather_unfold_next_token_pairs,
)


@pytest.mark.parametrize("block_size", [1, 2, 5, 12])
def test_indexed_and_unfold_gathers_match_at_boundaries(block_size):
    tokens = torch.arange(25, dtype=torch.long)
    starts = torch.tensor([0, len(tokens) - block_size - 1, 1, 1])
    expected = gather_next_token_pairs(tokens, starts, block_size=block_size)
    result = gather_unfold_next_token_pairs(tokens, starts, block_size=block_size)
    assert all(torch.equal(x, y) for x, y in zip(expected, result))


def test_unfold_window_table_is_a_view_not_corpus_squared_storage():
    tokens = torch.arange(101, dtype=torch.long)
    view = tokens.unfold(0, 9, 1)
    assert view.shape == (93, 9)
    assert view.untyped_storage().data_ptr() == tokens.untyped_storage().data_ptr()
    assert view.untyped_storage().nbytes() == tokens.untyped_storage().nbytes()


@pytest.mark.parametrize("invalid", [
    torch.tensor([], dtype=torch.long),
    torch.tensor([1], dtype=torch.int32),
    torch.tensor([-1]),
    torch.tensor([19]),
])
def test_invalid_start_vectors_are_rejected(invalid):
    with pytest.raises(ValueError, match="starts"):
        gather_unfold_next_token_pairs(
            torch.arange(20), invalid, block_size=3,
        )


def test_short_stream_and_wrong_dtype_are_rejected():
    with pytest.raises(ValueError, match="block_size"):
        gather_unfold_next_token_pairs(
            torch.arange(3), torch.tensor([0]), block_size=3,
        )
    with pytest.raises(ValueError, match="flat torch.long"):
        gather_unfold_next_token_pairs(
            torch.arange(20, dtype=torch.int32),
            torch.tensor([0]), block_size=3,
        )
