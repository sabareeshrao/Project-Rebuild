import math

import pytest
import torch

from rebuild_gpt.sampling import scale_logits, validate_temperature


def test_temperature_scales_logits_without_changing_their_shape():
    logits = torch.tensor([[0.0, 1.0, 3.0], [3.0, 1.0, 0.0]])
    assert torch.equal(scale_logits(logits, 1.0), logits)
    assert torch.allclose(scale_logits(logits, 2.0), logits / 2.0)
    assert torch.allclose(scale_logits(logits, 0.5), logits * 2.0)


def test_low_temperature_sharpens_high_probability_candidate():
    logits = torch.tensor([[0.0, 1.0, 3.0]])
    cold = scale_logits(logits, 0.5).softmax(dim=-1)
    warm = scale_logits(logits, 2.0).softmax(dim=-1)
    assert cold[0, -1] > warm[0, -1]
    assert math.isclose(cold.sum().item(), 1.0, abs_tol=1e-6)


@pytest.mark.parametrize("bad", [0, -1, -0.01, float("inf"), float("nan"), True, None])
def test_temperature_rejects_invalid_values(bad):
    with pytest.raises(ValueError, match="temperature"):
        validate_temperature(bad)


def test_temperature_requires_valid_logits():
    with pytest.raises(ValueError, match="logits"):
        scale_logits(torch.tensor([1, 2, 3]), 1.0)
