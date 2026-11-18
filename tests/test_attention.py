import math

import pytest
import torch

from rebuild_gpt.attention import CausalAttentionWeights, CausalSelfAttentionHead, ValueProjection, weighted_sum_values, QueryKeyProjection, pairwise_dot_scores, scaled_dot_scores, causal_mask_scores


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


def test_value_projection_shapes_and_gradients():
    layer = ValueProjection(n_embd=5, head_size=3)
    hidden = torch.randn(2, 4, 5, requires_grad=True)
    values = layer(hidden)
    assert values.shape == (2, 4, 3)
    values.sum().backward()
    assert hidden.grad is not None
    assert layer.value.weight.grad is not None


def test_value_projection_rejects_invalid_hidden():
    layer = ValueProjection(5, 3)
    with pytest.raises(ValueError, match="n_embd"):
        layer(torch.randn(2, 4, 4))
    with pytest.raises(ValueError, match="n_embd"):
        layer(torch.randint(0, 5, (2, 4, 5)))


def test_weighted_values_match_hand_calculation():
    weights = torch.tensor([[[1., 0.], [0.25, 0.75]]])
    values = torch.tensor([[[2., 0.], [0., 4.]]])
    mixed = weighted_sum_values(weights, values)
    assert torch.allclose(mixed, torch.tensor([[[2., 0.], [0.5, 3.]]]))


def test_weighted_sum_is_batched_differentiable_matrix_multiplication():
    weights = torch.randn(2, 3, 4, requires_grad=True)
    values = torch.randn(2, 4, 5, requires_grad=True)
    mixed = weighted_sum_values(weights, values)
    assert mixed.shape == (2, 3, 5)
    assert torch.allclose(mixed, weights @ values)
    mixed.sum().backward()
    assert weights.grad is not None and values.grad is not None


@pytest.mark.parametrize("weights,values", [
    (torch.randn(1, 2), torch.randn(1, 2, 3)),
    (torch.randn(1, 2, 4), torch.randn(1, 3, 5)),
    (torch.randn(2, 3, 2), torch.randn(1, 2, 2)),
])
def test_weighted_sum_rejects_dimension_mismatch(weights, values):
    with pytest.raises(ValueError, match="B,S"):
        weighted_sum_values(weights, values)


def test_single_head_matches_direct_pytorch_reference():
    torch.manual_seed(0)
    head = CausalSelfAttentionHead(n_embd=6, head_size=3)
    hidden = torch.randn(2, 4, 6, requires_grad=True)
    actual = head(hidden)
    q, k = head.attention_weights.projections(hidden)
    v = head.values(hidden)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(3)
    future = torch.ones(4, 4, dtype=torch.bool).triu(1)
    reference = scores.masked_fill(future, float("-inf")).softmax(dim=-1) @ v
    assert actual.shape == (2, 4, 3)
    assert torch.allclose(actual, reference, atol=1e-6)
    actual.sum().backward()
    assert hidden.grad is not None
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_single_head_prefix_outputs_do_not_depend_on_future():
    torch.manual_seed(1)
    head = CausalSelfAttentionHead(4, 2)
    hidden = torch.randn(2, 5, 4)
    changed = hidden.clone()
    changed[:, 3:, :] += 100
    assert torch.allclose(head(hidden)[:, :3], head(changed)[:, :3], atol=1e-6)
    assert torch.allclose(head(hidden)[:, :3], head(hidden[:, :3]), atol=1e-6)
