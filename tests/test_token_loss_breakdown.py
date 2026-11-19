"""Ignored target slots must be distinct from zero-loss observations."""

import pytest
import torch

from rebuild_gpt.token_loss_breakdown import inspect_token_losses


def test_masked_positions_are_none_and_mean_weights_each_token():
    logits = torch.tensor([
        [[2., 0.], [4., -2.], [0., 5.]],
        [[0., 3.], [1., -1.], [3., 0.]],
    ])
    labels = torch.tensor([[0, -1, -1], [1, 0, -1]])
    result = inspect_token_losses(logits, labels)
    assert result["supervised_positions"] == 3
    assert result["ignored_positions"] == 3
    assert result["row_supervised_counts"] == [1, 2]
    assert result["per_position_nll"][0][1:] == [None, None]
    assert result["per_position_nll"][1][-1] is None
    observed = [x for row in result["per_position_nll"] for x in row if x is not None]
    assert result["mean_nll"] == pytest.approx(sum(observed) / 3)
    assert result["mean_cross_entropy_error"] < 2e-6


def test_entirely_ignored_row_has_no_defined_mean():
    logits = torch.zeros(2, 2, 4)
    labels = torch.tensor([[-1, -1], [0, 1]])
    result = inspect_token_losses(logits, labels)
    assert result["row_mean_nll"][0] is None
    assert result["row_supervised_counts"] == [0, 2]
    assert result["mean_nll"] == pytest.approx(torch.tensor(4.).log().item())


def test_masked_logits_do_not_affect_loss_breakdown():
    logits = torch.zeros(1, 3, 3)
    targets = torch.tensor([[0, -1, 2]])
    baseline = inspect_token_losses(logits, targets)
    changed = logits.clone()
    changed[0, 1, :] = torch.tensor([100., -100., 50.])
    report = inspect_token_losses(changed, targets)
    assert report["mean_nll"] == pytest.approx(baseline["mean_nll"])
    assert report["per_position_nll"][0][1] is None


@pytest.mark.parametrize("targets", [
    torch.tensor([[-1, -1]]),
    torch.tensor([[0, 9]]),
    torch.tensor([[0., -1.]]),
])
def test_invalid_mask_or_target_rejected(targets):
    with pytest.raises(ValueError):
        inspect_token_losses(torch.zeros(1, 2, 3), targets)
