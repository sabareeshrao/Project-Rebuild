import math

import pytest
import torch

from rebuild_gpt.sampling import filter_top_k, scale_logits, validate_temperature, validate_top_k


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


def test_top_k_keeps_only_largest_candidates():
    logits = torch.tensor([[1., 5., 0., 3.], [4., 0., 2., 1.]])
    before = logits.clone()
    limited = filter_top_k(logits, 2)
    assert torch.equal(logits, before)
    assert limited.isneginf().sum().item() == 4
    assert torch.equal(limited.isfinite(), torch.tensor([[False, True, False, True], [True, False, True, False]]))


def test_top_k_ties_at_threshold_are_kept_like_upstream():
    logits = torch.tensor([[2., 2., 2., 0.]])
    result = filter_top_k(logits, 1)
    assert torch.equal(result.isfinite(), torch.tensor([[True, True, True, False]]))


def test_top_k_none_and_beyond_vocabulary_keep_all_candidates():
    logits = torch.tensor([[1., 2., 3.]])
    assert torch.equal(filter_top_k(logits, None), logits)
    assert torch.equal(filter_top_k(logits, 100), logits)


@pytest.mark.parametrize("bad", [0, -2, 1.5, True, "2"])
def test_invalid_top_k_rejected(bad):
    with pytest.raises(ValueError, match="top_k"):
        validate_top_k(bad)
