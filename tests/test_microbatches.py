"""Verify that splitting retains order, all examples and tensor storage."""

import pytest
import torch

from rebuild_gpt.microbatches import microbatch_sizes, split_microbatches


def test_nondivisible_batch_has_small_last_chunk():
    assert microbatch_sizes(10, 4) == (4, 4, 2)
    assert microbatch_sizes(5, 5) == (5,)
    assert microbatch_sizes(3, 10) == (3,)


def test_split_preserves_order_and_views_of_one_logical_batch():
    x = torch.arange(30).reshape(5, 6)
    y = torch.arange(20).reshape(5, 4)
    chunks = split_microbatches(x, y, microbatch_size=2)
    assert [a.size(0) for a, _ in chunks] == [2, 2, 1]
    assert torch.equal(torch.cat([a for a, _ in chunks]), x)
    assert torch.equal(torch.cat([b for _, b in chunks]), y)
    assert chunks[0][0].data_ptr() == x.data_ptr()
    assert chunks[0][1].data_ptr() == y.data_ptr()


def test_classifier_targets_may_have_different_trailing_shape():
    x = torch.ones(3, 8)
    y = torch.tensor([0, 1, 2])
    chunks = split_microbatches(x, y, microbatch_size=2)
    assert [b.shape for _, b in chunks] == [(2,), (1,)]


@pytest.mark.parametrize("total,micro", [
    (0, 2), (-2, 1), (3, 0), (3, -1), (True, 2),
    (3, False), (4.5, 2), (3, "2"),
])
def test_invalid_size_is_rejected(total, micro):
    with pytest.raises(ValueError, match="positive integer"):
        microbatch_sizes(total, micro)


@pytest.mark.parametrize("x,y", [
    (torch.ones(2, 4), torch.ones(3)),
    (torch.ones(0, 4), torch.ones(0)),
    (torch.tensor(1), torch.tensor(1)),
])
def test_split_rejects_invalid_batch_axes(x, y):
    with pytest.raises(ValueError, match="batch axis"):
        split_microbatches(x, y, microbatch_size=2)
