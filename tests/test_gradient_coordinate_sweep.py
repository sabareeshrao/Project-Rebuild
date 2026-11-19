"""A single correct embedding derivative need not imply all gradients work."""

import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.gradient_coordinate_sweep import check_gradient_coordinates


def _model():
    torch.manual_seed(17)
    return GPTLanguageModel(GPTConfig(
        9, 5, 8, 2, 1, attention_impl="fused",
    )).double()


def _batch():
    return (torch.tensor([[0, 1, 2, 3]]),
            torch.tensor([[1, 2, 3, -1]]))


def test_token_and_position_embedding_gradients_match_finite_differences():
    model = _model()
    model.train()
    weights_before = {
        name: tensor.detach().clone() for name, tensor in model.named_parameters()
    }
    report = check_gradient_coordinates(model, *_batch())
    assert report["status"] == "passed"
    assert report["checked_coordinates"] == 3
    assert report["checked_parameter_names"] == [
        "positions.lookup.weight", "tokens.lookup.weight",
    ]
    assert report["max_absolute_error"] < 1e-5
    assert model.training
    assert all(tensor.grad is None for tensor in model.parameters())
    assert all(torch.equal(weights_before[name], tensor)
               for name, tensor in model.named_parameters())


def test_corrupted_position_gradient_is_detected_and_weight_restored():
    model = _model()
    saved = model.positions.lookup.weight.detach().clone()
    hook = model.positions.lookup.weight.register_hook(lambda grad: grad + 0.25)
    try:
        with pytest.raises(AssertionError, match="finite difference mismatch"):
            check_gradient_coordinates(model, *_batch())
    finally:
        hook.remove()
    assert torch.equal(saved, model.positions.lookup.weight)
    assert all(param.grad is None for param in model.parameters())


@pytest.mark.parametrize("coordinates", [
    (), (("tokens.lookup.weight", (0, 0)),),
    (("tokens.lookup.weight", (0, 0)), ("tokens.lookup.weight", (0, 0))),
    ("tokens.lookup.weight", "positions.lookup.weight"),
])
def test_invalid_coordinate_plan_rejected(coordinates):
    with pytest.raises(ValueError, match="coordinates|distinct"):
        check_gradient_coordinates(_model(), *_batch(), coordinates=coordinates)
