"""The two uses of tied embeddings add their gradients into one tensor."""

import pytest
import torch
from torch import nn

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.tied_gradient_decomposition import check_tied_gradient_decomposition


def _model():
    return GPTLanguageModel(GPTConfig(
        11, 5, 8, 2, 1, dropout=0.2, attention_impl="fused",
    ))


def _batch():
    return (torch.tensor([[0, 1, 2, 3], [4, 3, 2, 1]]),
            torch.tensor([[1, 2, 3, 4], [3, 2, 1, 0]]))


def test_shared_gradient_equals_sum_of_untied_contributions():
    model = _model()
    model.train()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    report = check_tied_gradient_decomposition(model, *_batch())
    assert report["status"] == "matched"
    assert report["shared_gradient_error"] < 2e-5
    assert report["input_contribution_norm"] > 0
    assert report["output_contribution_norm"] > 0
    assert report["checked_other_parameters"] > 0
    assert model.training
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(a, p) for a, p in zip(before, model.parameters()))


def test_equal_but_separately_allocated_weights_are_not_tied():
    model = _model()
    model.head.linear.weight = nn.Parameter(
        model.tokens.lookup.weight.detach().clone()
    )
    with pytest.raises(AssertionError, match="share one Parameter"):
        check_tied_gradient_decomposition(model, *_batch())


def test_invalid_targets_do_not_start_backward():
    model = _model()
    targets = torch.tensor([[1, 2, -1, 4], [3, 2, 1, 0]])
    with pytest.raises(ValueError, match="real vocabulary"):
        check_tied_gradient_decomposition(model, _batch()[0], targets)
    assert all(parameter.grad is None for parameter in model.parameters())
