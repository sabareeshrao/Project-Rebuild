"""AdamW one-step equivalence does not imply matching whole training runs."""

import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.training_parity import compare_one_adamw_step
from test_training_parity_comparison import RenamedLocalReference


def _models():
    model = GPTLanguageModel(GPTConfig(7, 4, 8, 2, 1, attention_impl="fused"))
    return model, RenamedLocalReference(model)


def test_same_gradients_and_optimizer_produce_same_updated_weights():
    local, reference = _models()
    x = torch.tensor([[0, 1, 2, 3]])
    y = torch.tensor([[1, 2, 3, -1]])
    result = compare_one_adamw_step(local, reference, x, y)
    assert result.matched_gradients > 0
    assert result.matched_parameter_tensors >= result.matched_gradients
    assert result.max_gradient_error < 1e-6
    assert result.max_updated_weight_error < 1e-6
    assert result.largest_local_parameter_change > 0


@pytest.mark.parametrize("changes", [
    {"learning_rate": 0}, {"learning_rate": float("inf")},
    {"weight_decay": -0.5}, {"weight_decay": float("nan")},
])
def test_optimizer_settings_are_validated(changes):
    local, reference = _models()
    x, y = torch.tensor([[0, 1]]), torch.tensor([[1, -1]])
    with pytest.raises(ValueError):
        compare_one_adamw_step(local, reference, x, y, **changes)
