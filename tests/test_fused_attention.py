import pytest
import torch
from torch.nn import functional as F

from rebuild_gpt.fused_attention import FusedCausalSelfAttention
from rebuild_gpt.multihead import MultiHeadAttention


def test_fused_attention_agrees_with_pytorch_causal_reference():
    torch.manual_seed(7)
    attention = FusedCausalSelfAttention(n_embd=8, n_head=2)
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    q, k, v = attention.qkv(hidden)
    reference = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0)
    reference = attention.c_proj(reference.transpose(1, 2).contiguous().view(2, 5, 8))
    actual = attention(hidden)
    assert actual.shape == (2, 5, 8)
    assert torch.allclose(actual, reference, atol=1e-6, rtol=1e-5)
    actual.square().sum().backward()
    assert hidden.grad is not None
    assert all(param.grad is not None for param in attention.parameters())


def test_fused_attention_matches_educational_heads_with_same_weights():
    torch.manual_seed(9)
    legacy = MultiHeadAttention(n_embd=8, n_head=2, dropout=0.0)
    fused = FusedCausalSelfAttention(n_embd=8, n_head=2, dropout=0.0)
    qs, ks, vs = [], [], []
    for head in legacy.heads.heads:
        qs.append(head.attention_weights.projections.query.weight)
        ks.append(head.attention_weights.projections.key.weight)
        vs.append(head.values.value.weight)
    with torch.no_grad():
        fused.qkv.c_attn.weight.copy_(torch.cat([torch.cat(qs), torch.cat(ks), torch.cat(vs)]))
        fused.qkv.c_attn.bias.zero_()
        fused.c_proj.weight.copy_(legacy.projection.weight)
        fused.c_proj.bias.copy_(legacy.projection.bias)
    hidden = torch.randn(2, 5, 8)
    assert torch.allclose(fused(hidden), legacy(hidden), atol=1e-6, rtol=1e-5)


def test_fused_attention_is_causal_across_different_prefix_lengths():
    torch.manual_seed(11)
    attention = FusedCausalSelfAttention(12, 3)
    attention.eval()
    hidden = torch.randn(2, 6, 12)
    changed = hidden.clone()
    changed[:, 3:, :] *= -40
    with torch.no_grad():
        assert torch.allclose(attention(hidden)[:, :3], attention(changed)[:, :3], atol=1e-5)
        assert torch.allclose(attention(hidden)[:, :3], attention(hidden[:, :3]), atol=1e-5)


def test_fused_dropout_is_disabled_in_evaluation():
    layer = FusedCausalSelfAttention(8, 2, dropout=1.0)
    hidden = torch.randn(1, 4, 8)
    layer.train()
    assert torch.count_nonzero(layer(hidden)) == 0
    layer.eval()
    assert torch.count_nonzero(layer(hidden)) > 0


@pytest.mark.parametrize("rate", [-0.1, 1.1, float("nan")])
def test_fused_attention_rejects_invalid_dropout(rate):
    with pytest.raises(ValueError, match="dropout"):
        FusedCausalSelfAttention(8, 2, dropout=rate)
