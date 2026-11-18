import pytest
import torch

from rebuild_gpt.attention import QueryKeyProjection, pairwise_dot_scores


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
