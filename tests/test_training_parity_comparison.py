"""Small fake-upstream adapter tests orchestration without claiming genuine parity."""

from copy import deepcopy

import pytest
import torch
from torch import nn

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.nanogpt_keys import (
    export_nanogpt_state_dict, load_nanogpt_state_dict, upstream_key,
)
from rebuild_gpt.training_parity import compare_gradients_against_upstream


class RenamedLocalReference(nn.Module):
    """Local-only test double: never a substitute for real upstream testing."""

    def __init__(self, model):
        super().__init__()
        self.inner = deepcopy(model)

    def forward(self, x, y):
        return self.inner.forward_nanogpt(x, y)

    def state_dict(self, *args, **kwargs):
        return export_nanogpt_state_dict(self.inner)

    def load_state_dict(self, state, strict=True):
        load_nanogpt_state_dict(self.inner, state)

    def named_parameters(self, *args, **kwargs):
        for name, parameter in self.inner.named_parameters(*args, **kwargs):
            yield upstream_key(name), parameter


def _test_models():
    cfg = GPTConfig(7, 4, 8, 2, 1, attention_impl="fused", dropout=0.1)
    ours = GPTLanguageModel(cfg)
    return ours, RenamedLocalReference(ours)


def test_matching_reference_produces_equal_gradients_and_restores_modes():
    torch.manual_seed(9)
    ours, reference = _test_models()
    ours.train()
    reference.train()
    x = torch.tensor([[0, 1, 2, 3]])
    y = torch.tensor([[1, 2, 3, -1]])
    report = compare_gradients_against_upstream(ours, reference, x, y)
    assert report.matched_tensors > 0
    assert report.max_absolute_error == pytest.approx(0.0, abs=1e-8)
    assert ours.training and reference.training


def test_reference_gradient_key_disagreement_is_rejected():
    ours, reference = _test_models()
    reference.named_parameters = lambda *a, **k: iter([])
    x, y = torch.tensor([[0, 1]]), torch.tensor([[1, -1]])
    with pytest.raises(ValueError, match="Gradient key mismatch"):
        compare_gradients_against_upstream(ours, reference, x, y)


def test_invalid_input_shapes_fail_cleanly():
    ours, reference = _test_models()
    with pytest.raises(ValueError, match="\[B,T\]"):
        compare_gradients_against_upstream(
            ours, reference, torch.tensor([0, 1]), torch.tensor([1, 2]),
        )
