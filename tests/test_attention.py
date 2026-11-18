import math

import pytest
import torch

from rebuild_gpt.attention import CausalAttentionWeights, QueryKeyProjection, pairwise_dot_scores, scaled_dot_scores, causal_mask_scores


def test_pairwise_dot_scores_match_hand_calculated_values():
    q = torch.tensor([[[1., 0.], [0., 1.]]], requires_grad=True)
    k = torch.tensor([[[1., 0.], [1., 1.], [0., 1.]]], requires_grad=True)
    scores = pairwise_dot_scores(q, k)
    assert scores.shape == (1, 2, 3)
    assert torch.equal(scores[0], torch.tensor([[1., 1., 0.], [0., 1., 1.]]))
    scores.sum().backward()
    assert q.grad is not None and k.grad is not None


@pytest.mark.parametrize("q,k", [
    (torch.randn(2, 3), torch.randn(2, 3)),
    (torch.randn(1, 2, 3), torch.randn(2, 2, 3)),
    (torch.randn(1, 2, 3), torch.randn(1, 2, 4)),
    (torch.zeros(1, 0, 3), torch.zeros(1, 1, 3)),
])
def test_pairwise_dot_scores_reject_bad_shapes(q, k):
    with pytest.raises(ValueError, match="queries"):
        pairwise_dot_scores(q, k)


def test_query_key_projection_shapes_and_gradients():
    torch.manual_seed(2)
    layer = QueryKeyProjection(n_embd=5, head_size=3)
    hidden = torch.randn(2, 4, 5, requires_grad=True)
    q, k = layer(hidden)
    assert q.shape == (2, 4, 3) and k.shape == (2, 4, 3)
    assert pairwise_dot_scores(q, k).shape == (2, 4, 4)
    (q.sum() + k.sum()).backward()
    assert hidden.grad is not None
    assert layer.query.weight.grad is not None
    assert layer.key.weight.grad is not None


def test_query_key_projection_does_not_share_weights():
    layer = QueryKeyProjection(4, 2)
    assert layer.query.weight is not layer.key.weight


def test_query_key_projection_rejects_bad_hidden_shape():
    layer = QueryKeyProjection(4, 2)
    with pytest.raises(ValueError, match="n_embd"):
        layer(torch.randn(2, 3, 5))


def test_scaling_reduces_dot_products_by_sqrt_width():
    queries = torch.tensor([[[2., 0.]]])
    keys = torch.tensor([[[1., 0.], [2., 1.]]])
    raw = pairwise_dot_scores(queries, keys)
    scaled = scaled_dot_scores(queries, keys)
    assert torch.allclose(scaled, raw / math.sqrt(2))


def test_scaling_matches_pytorch_float_and_can_backpropagate():
    q = torch.randn(2, 3, 4, requires_grad=True)
    k = torch.randn(2, 3, 4, requires_grad=True)
    scored = scaled_dot_scores(q, k)
    assert torch.allclose(scored, torch.matmul(q, k.transpose(-2, -1)) / 2)
    scored.sum().backward()
    assert q.grad is not None and k.grad is not None


def test_causal_mask_preserves_past_and_self_but_blocks_future():
    scores = torch.arange(9, dtype=torch.float32).reshape(1, 3, 3)
    masked = causal_mask_scores(scores)
    expected = torch.tensor([[[0., -torch.inf, -torch.inf],
                              [3., 4., -torch.inf],
                              [6., 7., 8.]]])
    assert torch.equal(masked, expected)
    assert torch.equal(scores, torch.arange(9, dtype=torch.float32).reshape(1, 3, 3))


@pytest.mark.parametrize("shape", [(3, 3), (1, 2, 3), (1, 0, 0)])
def test_causal_mask_rejects_invalid_shapes(shape):
    with pytest.raises(ValueError, match="square"):
        causal_mask_scores(torch.zeros(shape))


def test_attention_weights_are_normalized_causal_and_batched():
    torch.manual_seed(4)
    layer = CausalAttentionWeights(n_embd=6, head_size=3)
    hidden = torch.randn(2, 4, 6)
    weights = layer(hidden)
    assert weights.shape == (2, 4, 4)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4))
    assert torch.count_nonzero(torch.triu(weights, diagonal=1)) == 0
    assert torch.equal(weights[:, 0, 0], torch.ones(2))


def test_attention_weights_for_prefix_ignore_future_tokens():
    torch.manual_seed(6)
    layer = CausalAttentionWeights(n_embd=4, head_size=2)
    hidden = torch.randn(1, 5, 4)
    other = hidden.clone()
    other[:, 3:, :] = torch.randn_like(other[:, 3:, :])
    assert torch.allclose(layer(hidden)[:, :3, :3], layer(other)[:, :3, :3])


def test_attention_weights_backpropagate_to_query_and_key():
    torch.manual_seed(8)
    layer = CausalAttentionWeights(4, 3)
    hidden = torch.randn(1, 4, 4, requires_grad=True)
    weights = layer(hidden)
    weights[:, -1, 0].sum().backward()
    assert hidden.grad is not None
    assert layer.projections.query.weight.grad is not None
    assert layer.projections.key.weight.grad is not None
