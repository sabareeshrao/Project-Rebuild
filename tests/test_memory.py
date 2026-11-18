import pytest
import torch

from rebuild_gpt.memory import estimate_model_memory, mebibytes


class SharedWeights(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.output = torch.nn.Linear(4, 8, bias=False)
        self.output.weight = self.embedding.weight
        self.frozen = torch.nn.Parameter(torch.ones(3), requires_grad=False)
        self.register_buffer("scale", torch.ones(2))


def test_tied_parameters_are_only_counted_once():
    model = SharedWeights()
    size = estimate_model_memory(model)
    assert size.trainable_parameters == 8 * 4
    assert size.parameter_bytes == (8 * 4 + 3) * 4
    assert size.buffer_bytes == 2 * 4
    assert size.gradient_bytes == (8 * 4) * 4
    assert size.adamw_moment_bytes == 2 * (8 * 4) * 4
    assert size.estimated_training_bytes == (35 + 2 + 32 + 64) * 4


def test_unfrozen_parameters_have_gradient_and_adamw_budget():
    model = torch.nn.Linear(3, 2, bias=True)
    size = estimate_model_memory(model)
    assert size.parameter_bytes == 8 * 4
    assert size.gradient_bytes == 8 * 4
    assert size.adamw_moment_bytes == 16 * 4


def test_memory_budget_is_not_measured_peak_memory():
    model = SharedWeights()
    assert estimate_model_memory(model).estimated_training_bytes > 0
    assert mebibytes(1024 * 1024) == 1
    with pytest.raises(ValueError, match="nonnegative"):
        mebibytes(-1)


def test_estimator_rejects_non_modules():
    with pytest.raises(TypeError, match="nn.Module"):
        estimate_model_memory(object())
