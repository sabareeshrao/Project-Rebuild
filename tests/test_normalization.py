import pytest
import torch

from rebuild_gpt.normalization import normalize_features


def test_normalize_features_zero_mean_and_unit_variance():
    x = torch.tensor([[[1., 2., 3., 4.], [5., 7., 9., 11.]]])
    out = normalize_features(x)
    assert out.shape == x.shape
    assert torch.allclose(out.mean(-1), torch.zeros(1, 2), atol=1e-6)
    assert torch.allclose(out.var(-1, unbiased=False), torch.ones(1, 2), atol=1e-4)


def test_normalization_does_not_mix_tokens_or_batch_rows():
    x = torch.randn(2, 4, 6)
    changed = x.clone()
    changed[:, 3] = 100.0
    assert torch.allclose(normalize_features(x)[:, :3], normalize_features(changed)[:, :3])


def test_constant_vectors_normalize_to_zero_without_nan():
    assert torch.equal(normalize_features(torch.full((1, 2, 4), 3.0)), torch.zeros(1, 2, 4))


@pytest.mark.parametrize("eps", [0.0, -0.1, float("nan")])
def test_invalid_epsilon_is_rejected(eps):
    with pytest.raises(ValueError, match="eps"):
        normalize_features(torch.randn(1, 2, 3), eps=eps)
