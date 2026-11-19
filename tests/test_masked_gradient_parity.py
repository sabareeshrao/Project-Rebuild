"""A target ignored by cross-entropy must also be ignored by backward."""

import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.masked_gradient_parity import check_masked_target_gradients


def _model(cls=GPTLanguageModel):
    return cls(GPTConfig(
        11, 5, 8, 2, 1, dropout=0.2, attention_impl="fused",
    ))


def _inputs():
    return (torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]]),
            torch.tensor([[1, -1, 3, -1], [2, 1, -1, 1]]))


def test_masked_gradients_agree_with_supervised_subset():
    model = _model()
    model.train()
    initial = [p.detach().clone() for p in model.parameters()]
    report = check_masked_target_gradients(model, *_inputs())
    assert report["status"] == "matched"
    assert report["supervised_positions"] == 5
    assert report["ignored_positions"] == 3
    assert report["matched_parameter_gradients"] > 0
    assert report["max_gradient_error"] < 2e-5
    assert model.training
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(before, after)
               for before, after in zip(initial, model.parameters()))


class WrongLossGPT(GPTLanguageModel):
    def forward_nanogpt(self, token_ids, targets=None):
        logits, loss = super().forward_nanogpt(token_ids, targets)
        return logits, 1.3 * loss if loss is not None else None


def test_stealthily_scaled_loss_and_gradients_are_detected():
    with pytest.raises(AssertionError, match="loss differs"):
        check_masked_target_gradients(_model(WrongLossGPT), *_inputs())


@pytest.mark.parametrize("labels", [
    torch.tensor([[1, 2, 3, 4]]),
    torch.tensor([[-1, -1, -1, -1], [-1, -1, -1, -1]]),
    torch.tensor([[1, 2, 3, 4], [3, 2, 1, 0]]),
    torch.tensor([[1., -1., 3., -1.], [2., 1., -1., 1.]]),
    torch.tensor([[1, -2, 3, -1], [2, 1, -1, 1]]),
])
def test_invalid_mask_patterns_are_rejected(labels):
    with pytest.raises(ValueError):
        check_masked_target_gradients(_model(), _inputs()[0], labels)
