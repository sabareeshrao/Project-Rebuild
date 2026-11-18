import pytest
import torch

from rebuild_gpt.probability import stable_softmax


def test_equal_logits_produce_uniform_probabilities():
    logits = torch.zeros(2, 3, 4)
    probs = stable_softmax(logits)
    assert probs.shape == logits.shape
    assert torch.allclose(probs, torch.full_like(probs, 0.25))
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2, 3))


def test_stable_softmax_handles_large_scores():
    logits = torch.tensor([[1000.0, 1001.0, 1002.0]])
    probs = stable_softmax(logits)
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs, torch.softmax(logits, dim=-1))


def test_softmax_rejects_integer_logits():
    with pytest.raises(ValueError, match="floating"):
        stable_softmax(torch.tensor([1, 2, 3]))
