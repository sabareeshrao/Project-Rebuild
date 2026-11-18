import pytest
import torch
from torch import nn

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.optimizers import configure_adamw


def test_optimizer_groups_unique_trainable_parameters_and_tied_weights():
    model = GPTLanguageModel(GPTConfig(11, 4, 8, 2, 2, attention_impl="fused"))
    opt = configure_adamw(model, learning_rate=0.001, weight_decay=0.12)
    assert len(opt.param_groups) == 2
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == pytest.approx(0.12)
    assert no_decay["weight_decay"] == 0
    assert all(p.ndim >= 2 for p in decay["params"])
    assert all(p.ndim < 2 for p in no_decay["params"])
    ids = [id(p) for group in opt.param_groups for p in group["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    tied_id = id(model.tokens.lookup.weight)
    assert ids.count(tied_id) == 1
    assert tied_id in [id(p) for p in decay["params"]]
    assert id(model.final_norm.weight) in [id(p) for p in no_decay["params"]]


def test_weight_decay_only_applies_to_matrix_weights_with_zero_gradient():
    layer = nn.Linear(3, 2)
    with torch.no_grad():
        layer.weight.fill_(1.0)
        layer.bias.fill_(1.0)
    opt = configure_adamw(layer, learning_rate=0.1, weight_decay=0.2, betas=(0.9, 0.95))
    for p in layer.parameters():
        p.grad = torch.zeros_like(p)
    opt.step()
    torch.testing.assert_close(layer.weight, torch.full_like(layer.weight, 0.98))
    torch.testing.assert_close(layer.bias, torch.ones_like(layer.bias))


def test_frozen_parameters_are_excluded():
    layer = nn.Linear(3, 2)
    layer.bias.requires_grad_(False)
    opt = configure_adamw(layer, learning_rate=0.01)
    assert [p for g in opt.param_groups for p in g["params"]] == [layer.weight]


@pytest.mark.parametrize("options,match", [
    ({"learning_rate": 0}, "learning_rate"),
    ({"weight_decay": -0.1}, "weight_decay"),
    ({"weight_decay": float("inf")}, "weight_decay"),
    ({"betas": (0.9, 1.0)}, "betas"),
    ({"betas": (0.9,)}, "betas"),
    ({"device_type": "mps"}, "device_type"),
])
def test_adamw_group_configuration_validation(options, match):
    settings = {"learning_rate": 0.01}
    settings.update(options)
    with pytest.raises(ValueError, match=match):
        configure_adamw(nn.Linear(2, 2), **settings)
