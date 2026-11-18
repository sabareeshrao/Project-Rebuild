import torch
from torch.nn import functional as F

from rebuild_gpt.feedforward import FeedForward
from rebuild_gpt.initialization import initialize_gpt_weights
from rebuild_gpt.normalization import FeatureLayerNorm


def test_bias_free_layernorm_matches_pytorch():
    layer = FeatureLayerNorm(5, bias=False)
    inputs = torch.randn(2, 3, 5, requires_grad=True)
    assert layer.bias is None
    assert torch.allclose(
        layer(inputs),
        F.layer_norm(inputs, (5,), layer.weight, None),
        atol=1e-6,
    )
    layer(inputs).square().sum().backward()
    assert layer.weight.grad is not None
    assert inputs.grad is not None


def test_bias_free_feedforward_has_no_bias_parameters():
    mlp = FeedForward(6, bias=False)
    assert mlp.expand.bias is None and mlp.project.bias is None
    assert mlp(torch.randn(2, 4, 6)).shape == (2, 4, 6)
    initialize_gpt_weights(mlp, 2)


def test_bias_true_keeps_existing_default_behaviour():
    assert FeatureLayerNorm(4).bias is not None
    assert FeedForward(4).expand.bias is not None
