"""An independently measured slope checks that GPT backprop is plausible."""

import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.gradient_check import check_parameter_gradient


def _model():
    torch.manual_seed(31)
    return GPTLanguageModel(GPTConfig(
        9, 5, 8, 2, 1, attention_impl="fused",
    )).double()


def _batch():
    return torch.tensor([[0, 1, 2, 3]]), torch.tensor([[1, 2, 3, -1]])


def test_one_tied_embedding_weight_matches_numerical_derivative():
    model = _model()
    model.train()
    original = model.tokens.lookup.weight.detach().clone()
    x, y = _batch()
    result = check_parameter_gradient(model, x, y)
    assert result["parameter"] == "tokens.lookup.weight"
    assert result["index"] == [0, 0]
    assert result["absolute_error"] < 1e-5
    assert model.training
    assert torch.equal(model.tokens.lookup.weight, original)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_corrupted_backward_gradient_is_detected():
    model = _model()
    hook = model.tokens.lookup.weight.register_hook(lambda grad: grad * 2.0)
    try:
        with pytest.raises(AssertionError, match="finite difference mismatch"):
            check_parameter_gradient(model, *_batch())
    finally:
        hook.remove()


def test_float32_is_not_sufficient_for_this_strict_numerical_check():
    model = GPTLanguageModel(GPTConfig(9, 5, 8, 2, 1, attention_impl="fused"))
    with pytest.raises(ValueError, match="float64"):
        check_parameter_gradient(model, *_batch())


@pytest.mark.parametrize("kwargs", [
    {"epsilon": 0.0},
    {"epsilon": float("inf")},
    {"index": (100, 0)},
    {"index": (True, 0)},
    {"parameter_name": "not-a-weight"},
])
def test_invalid_gradient_check_configuration(kwargs):
    with pytest.raises(ValueError):
        check_parameter_gradient(_model(), *_batch(), **kwargs)
