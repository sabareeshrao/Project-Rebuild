"""The nanoGPT forward API must handle ignored targets and final logits."""

import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.model_invariants import check_forward_contract


def _model():
    return GPTLanguageModel(GPTConfig(
        11, 5, 8, 2, 1, dropout=0.15, attention_impl="fused",
    ))


def test_train_and_inference_contract_with_ignored_targets():
    model = _model()
    model.train()
    tokens = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    targets = torch.tensor([[1, 2, -1, -1], [2, 1, 0, -1]])
    result = check_forward_contract(model, tokens, targets)
    assert result["supervised_positions"] == 5
    assert result["ignored_positions"] == 3
    assert result["max_train_logit_error"] < 2e-5
    assert result["max_last_logit_error"] < 2e-5
    assert result["masked_loss_error"] < 2e-5
    assert model.training


class IncorrectInference(GPTLanguageModel):
    def forward_nanogpt(self, token_ids, targets=None):
        logits, loss = super().forward_nanogpt(token_ids, targets)
        if targets is None:
            return self(token_ids)[:, :1, :], None  # WRONG position
        return logits, loss


def test_wrong_inference_position_is_detected():
    model = IncorrectInference(GPTConfig(
        11, 5, 8, 2, 1, attention_impl="fused",
    ))
    with pytest.raises(AssertionError, match="forward contract"):
        check_forward_contract(
            model, torch.tensor([[0, 1, 2]]), torch.tensor([[1, 2, -1]]),
        )


class IncorrectMaskedLoss(GPTLanguageModel):
    def forward_nanogpt(self, token_ids, targets=None):
        logits, loss = super().forward_nanogpt(token_ids, targets)
        return logits, loss + 0.25 if loss is not None else None


def test_wrong_target_loss_is_detected():
    model = IncorrectMaskedLoss(GPTConfig(
        11, 5, 8, 2, 1, attention_impl="fused",
    ))
    with pytest.raises(AssertionError, match="forward contract"):
        check_forward_contract(
            model, torch.tensor([[0, 1, 2]]), torch.tensor([[1, 2, -1]]),
        )


@pytest.mark.parametrize("bad_targets", [
    torch.tensor([[1, 2]]),
    torch.tensor([[1., 2., 3.]]),
    torch.tensor([[-1, -1, -1]]),
    torch.tensor([[1, 2, 99]]),
    torch.tensor([[1, 2, -2]]),
])
def test_invalid_targets_are_rejected(bad_targets):
    with pytest.raises(ValueError):
        check_forward_contract(
            _model(), torch.tensor([[0, 1, 2]]), bad_targets,
        )
