import pytest
import torch

from rebuild_gpt.best_model import BestValidationModel


def test_best_snapshot_roundtrip_is_independent_of_saved_tensors():
    torch.manual_seed(7)
    model = torch.nn.Linear(3, 2)
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best = BestValidationModel()
    assert best.consider(model, 3, 0.8)
    state = best.state_dict()
    loaded = BestValidationModel()
    loaded.load_state_dict(state)
    with torch.no_grad():
        model.weight.add_(4)
    state["model_state"]["weight"].add_(3)
    loaded.restore(model)
    assert loaded.best_step == 3
    assert loaded.best_loss == 0.8
    assert all(torch.equal(model.state_dict()[k], v) for k, v in original.items())


def test_best_snapshot_can_be_uninitialized():
    state = BestValidationModel().state_dict()
    restored = BestValidationModel()
    restored.load_state_dict(state)
    assert restored.best_step is None
    with pytest.raises(ValueError, match="No best"):
        restored.restore(torch.nn.Linear(1, 1))


def test_best_snapshot_rejects_inconsistent_step():
    with pytest.raises(ValueError, match="invalid"):
        BestValidationModel().load_state_dict({
            "best_loss": -1, "best_step": 2, "model_state": {},
        })
