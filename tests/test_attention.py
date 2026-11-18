import pytest
import torch

from rebuild_gpt.attention import pairwise_dot_scores


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
