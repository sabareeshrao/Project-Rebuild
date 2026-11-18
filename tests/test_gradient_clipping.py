import pytest
import torch
from torch import nn

from rebuild_gpt.training import train_step


class OneWeightModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([10.0]))

    def loss(self, inputs, targets):
        return (self.weight * inputs.float()).square().mean()


def test_clipping_caps_gradient_before_optimizer_step():
    model = OneWeightModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    loss = train_step(model, optimizer, torch.ones(1, 1), torch.zeros(1, 1), max_grad_norm=0.1)
    assert loss == pytest.approx(100.0)
    assert model.weight.item() == pytest.approx(9.9, abs=1e-5)
    assert model.weight.grad.norm().item() <= 0.10001


def test_default_training_step_does_not_clip():
    model = OneWeightModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    train_step(model, optimizer, torch.ones(1, 1), torch.zeros(1, 1))
    assert model.weight.item() == pytest.approx(-10.0)


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True])
def test_bad_gradient_clipping_threshold_rejected(bad):
    model = OneWeightModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    with pytest.raises(ValueError, match="max_grad_norm"):
        train_step(model, optimizer, torch.ones(1, 1), torch.zeros(1, 1), max_grad_norm=bad)
