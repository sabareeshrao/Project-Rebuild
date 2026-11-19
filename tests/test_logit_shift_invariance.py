"""A constant logit offset is not a prediction change."""

import pytest
import torch

from rebuild_gpt.logit_shift_invariance import check_logit_shift_invariance


def test_probabilities_loss_and_gradient_unchanged_by_logit_offset():
    scores = torch.tensor([
        [[1., 0., -1.], [0., 2., 1.]],
        [[-1., 0., 1.], [3., 0., -2.]],
    ])
    targets = torch.tensor([[0, -1], [2, 0]])
    result = check_logit_shift_invariance(scores, targets, offset=3.25)
    assert result["status"] == "invariant"
    assert result["supervised_tokens"] == 3
    assert result["probability_error"] < 2e-6
    assert result["gradient_error"] < 2e-6


def test_negative_offset_also_preserves_float64():
    logits = torch.tensor([[[2., -3., 0.]]], dtype=torch.float64)
    result = check_logit_shift_invariance(
        logits, torch.tensor([[1]]), offset=-6.5,
    )
    assert result["status"] == "invariant"


@pytest.mark.parametrize("targets", [
    torch.tensor([[0, 1]]),
    torch.tensor([[-1, -1], [-1, -1]]),
    torch.tensor([[0, -2], [1, 0]]),
    torch.tensor([[0., -1.], [1., 2.]]),
])
def test_invalid_targets_are_rejected(targets):
    logits = torch.zeros(2, 2, 3)
    with pytest.raises(ValueError):
        check_logit_shift_invariance(logits, targets)


def test_nonfinite_offsets_cannot_claim_numerical_stability():
    with pytest.raises(ValueError, match="finite"):
        check_logit_shift_invariance(
            torch.zeros(1, 2, 3), torch.tensor([[0, 1]]),
            offset=float("inf"),
        )
