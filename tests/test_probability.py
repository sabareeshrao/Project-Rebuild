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


from rebuild_gpt.probability import next_token_cross_entropy


def test_uniform_cross_entropy_equals_log_vocab_size():
    logits = torch.zeros(2, 3, 4)
    targets = torch.tensor([[0, 1, 2], [3, 2, 1]])
    loss = next_token_cross_entropy(logits, targets)
    assert torch.allclose(loss, torch.log(torch.tensor(4.0)))


def test_cross_entropy_matches_pytorch_and_backpropagates():
    logits = torch.tensor([[[1.0, 2.0], [4.0, -2.0]]], requires_grad=True)
    targets = torch.tensor([[1, 0]])
    loss = next_token_cross_entropy(logits, targets)
    expected = torch.nn.functional.cross_entropy(logits.reshape(-1, 2), targets.reshape(-1))
    assert torch.allclose(loss, expected)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_loss_improves_if_correct_target_is_more_likely():
    target = torch.tensor([[1]])
    low = next_token_cross_entropy(torch.tensor([[[3.0, 0.0, 0.0]]]), target)
    high = next_token_cross_entropy(torch.tensor([[[0.0, 3.0, 0.0]]]), target)
    assert high < low


def test_cross_entropy_rejects_mismatched_targets():
    with pytest.raises(ValueError, match="shape"):
        next_token_cross_entropy(torch.zeros(1, 2, 3), torch.tensor([[0]]))
    with pytest.raises(ValueError, match="outside"):
        next_token_cross_entropy(torch.zeros(1, 2, 3), torch.tensor([[0, 3]]))
