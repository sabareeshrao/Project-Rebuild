import pytest
import torch

from rebuild_gpt.multihead import MultiHeadAttention, ParallelAttentionHeads


def test_parallel_heads_concatenate_independent_head_outputs():
    torch.manual_seed(6)
    layer = ParallelAttentionHeads(n_embd=8, n_head=3, head_size=2)
    hidden = torch.randn(2, 4, 8, requires_grad=True)
    outputs = [head(hidden) for head in layer.heads]
    actual = layer(hidden)
    assert actual.shape == (2, 4, 6)
    assert torch.allclose(actual, torch.cat(outputs, dim=-1))
    assert len(layer.heads) == 3
    assert layer.heads[0] is not layer.heads[1]
    actual.sum().backward()
    assert hidden.grad is not None
    assert all(param.grad is not None for param in layer.parameters())


def test_parallel_heads_preserve_causal_prefix_independence():
    torch.manual_seed(1)
    layer = ParallelAttentionHeads(6, 2, 3)
    hidden = torch.randn(2, 5, 6)
    changed = hidden.clone()
    changed[:, 3:] += 20
    assert torch.allclose(layer(hidden)[:, :3], layer(changed)[:, :3], atol=1e-6)


@pytest.mark.parametrize("config", [(0, 2, 3), (6, 0, 3), (6, 2, 0)])
def test_parallel_heads_require_positive_dimensions(config):
    with pytest.raises(ValueError, match="positive"):
        ParallelAttentionHeads(*config)


def test_multihead_output_projection_restores_model_width():
    torch.manual_seed(9)
    layer = MultiHeadAttention(n_embd=8, n_head=2)
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    concatenated = layer.heads(hidden)
    assert concatenated.shape == (2, 5, 8)
    actual = layer(hidden)
    assert actual.shape == (2, 5, 8)
    assert torch.allclose(actual, layer.projection(concatenated))
    actual.square().sum().backward()
    assert layer.projection.weight.grad is not None
    assert all(p.grad is not None for p in layer.heads.parameters())


@pytest.mark.parametrize("width,heads", [(0, 1), (4, 0), (7, 2), (2, 4)])
def test_multihead_rejects_incompatible_head_counts(width, heads):
    with pytest.raises(ValueError, match="divisible"):
        MultiHeadAttention(width, heads)


def test_multihead_projection_stays_causal():
    torch.manual_seed(11)
    layer = MultiHeadAttention(n_embd=8, n_head=4)
    hidden = torch.randn(1, 5, 8)
    changed = hidden.clone()
    changed[:, -2:] *= 50
    assert torch.allclose(layer(hidden)[:, :3], layer(changed)[:, :3], atol=1e-6)
