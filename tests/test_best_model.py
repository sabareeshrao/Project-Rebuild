import pytest
import torch
from torch import nn

from rebuild_gpt.best_model import BestValidationModel
from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel


def test_tracker_restores_old_weights_without_aliasing_future_updates():
    model = nn.Linear(3, 2)
    tracker = BestValidationModel()
    original = model.weight.detach().clone()
    assert tracker.consider(model, step=0, validation_loss=1.2)
    with torch.no_grad():
        model.weight.add_(3)
    assert not tracker.consider(model, step=1, validation_loss=1.4)
    assert not tracker.consider(model, step=2, validation_loss=1.2)
    assert tracker.best_step == 0 and tracker.best_loss == 1.2
    tracker.restore(model)
    assert torch.equal(model.weight, original)


def test_strictly_better_validation_loss_updates_snapshot():
    model = nn.Linear(2, 2)
    tracker = BestValidationModel()
    tracker.consider(model, 0, 4.0)
    with torch.no_grad():
        model.weight.fill_(9)
    assert tracker.consider(model, 4, 2.0)
    with torch.no_grad():
        model.weight.zero_()
    tracker.restore(model)
    assert torch.equal(model.weight, torch.full_like(model.weight, 9))
    assert tracker.best_step == 4


def test_restoring_gpt_preserves_tied_embeddings():
    model = GPTLanguageModel(GPTConfig(5, 4, 8, 2, 1, attention_impl="fused"))
    tracker = BestValidationModel()
    original = model.tokens.lookup.weight.detach().clone()
    tracker.consider(model, 1, 0.5)
    with torch.no_grad():
        model.tokens.lookup.weight.add_(1)
    tracker.restore(model)
    assert model.tokens.lookup.weight is model.head.linear.weight
    assert torch.equal(model.tokens.lookup.weight, original)


def test_cannot_restore_before_first_measurement():
    with pytest.raises(ValueError, match="No best model"):
        BestValidationModel().restore(nn.Linear(2, 2))


@pytest.mark.parametrize("step,score", [
    (-1, 1.0), (1.5, 1.0), (0, float("nan")),
    (0, float("inf")), (0, -0.1),
])
def test_tracker_rejects_invalid_measurements(step, score):
    with pytest.raises(ValueError):
        BestValidationModel().consider(nn.Linear(2, 2), step, score)
