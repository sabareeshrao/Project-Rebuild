"""Uneven masked microbatches require token counts, not equal row weights."""

import pytest
import torch

from rebuild_gpt.masked_microbatch_loss import check_masked_microbatch_equivalence


def _batch():
    logits = torch.tensor([
        [[4., 0.], [1., 2.], [2., 1.]],
        [[-3., 5.], [3., -1.], [5., -2.]],
    ])
    targets = torch.tensor([[0, -1, -1], [0, 1, 1]])
    return logits, targets


def test_weighting_by_valid_token_count_reproduces_loss_and_gradient():
    x, y = _batch()
    result = check_masked_microbatch_equivalence(x, y)
    assert result["status"] == "matched"
    assert result["microbatch_supervised_counts"] == [1, 3]
    assert result["total_supervised_positions"] == 4
    assert result["loss_error"] < 2e-6
    assert result["max_gradient_error"] < 2e-6
    assert result["naive_unweighted_mean_error"] > 0.1


def test_entirely_masked_microbatch_contributes_nothing():
    x, y = _batch()
    y = y.clone()
    y[0] = -1
    result = check_masked_microbatch_equivalence(x, y)
    assert result["microbatch_supervised_counts"] == [0, 3]
    assert result["max_gradient_error"] < 2e-6


def test_larger_batch_can_split_after_two_rows():
    x, y = _batch()
    x = torch.cat((x, x[:1]), dim=0)
    y = torch.cat((y, y[:1]), dim=0)
    result = check_masked_microbatch_equivalence(x, y, split_rows=2)
    assert result["microbatch_supervised_counts"] == [4, 1]


@pytest.mark.parametrize("split", [0, 2, True, 1.5])
def test_invalid_split_rejected_before_any_backward(split):
    with pytest.raises(ValueError, match="split_rows"):
        check_masked_microbatch_equivalence(*_batch(), split_rows=split)


def test_all_ignored_batch_cannot_claim_matching_gradients():
    x, y = _batch()
    with pytest.raises(ValueError, match="supervised"):
        check_masked_microbatch_equivalence(x, torch.full_like(y, -1))
