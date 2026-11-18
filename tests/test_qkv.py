import pytest
import torch

from rebuild_gpt.qkv import FusedQKVProjection


def test_fused_qkv_matches_linear_chunks_and_head_reshape():
    torch.manual_seed(2)
    module = FusedQKVProjection(n_embd=8, n_head=2)
    x = torch.randn(3, 5, 8, requires_grad=True)
    q, k, v = module(x)
    assert q.shape == k.shape == v.shape == (3, 2, 5, 4)
    raw = module.c_attn(x)
    manual = [chunk.reshape(3, 5, 2, 4).transpose(1, 2) for chunk in raw.chunk(3, dim=-1)]
    assert all(torch.equal(actual, expected) for actual, expected in zip((q, k, v), manual))
    (q.square().sum() + k.sum() + v.sum()).backward()
    assert module.c_attn.weight.grad is not None and x.grad is not None


def test_fused_qkv_bias_and_weight_layout():
    layer = FusedQKVProjection(12, 3)
    assert layer.c_attn.weight.shape == (36, 12)
    assert layer.c_attn.bias.shape == (36,)


@pytest.mark.parametrize("width,heads", [(0, 1), (8, 0), (7, 2)])
def test_fused_qkv_validates_config(width, heads):
    with pytest.raises(ValueError, match="divisible"):
        FusedQKVProjection(width, heads)


def test_fused_qkv_rejects_wrong_input_width():
    with pytest.raises(ValueError, match="n_embd"):
        FusedQKVProjection(8, 2)(torch.randn(1, 2, 6))
