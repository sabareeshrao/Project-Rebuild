import pytest
import torch

from rebuild_gpt.baselines import uniform_logits


def test_uniform_baseline_returns_one_score_per_vocab_token():
    inputs = torch.tensor([[0, 1, 2], [2, 1, 0]])
    logits = uniform_logits(inputs, vocab_size=4)
    assert logits.shape == (2, 3, 4)
    assert logits.dtype == torch.float32
    assert torch.count_nonzero(logits) == 0


def test_uniform_baseline_rejects_bad_inputs():
    with pytest.raises(ValueError, match="non-empty"):
        uniform_logits(torch.tensor([0, 1]), vocab_size=3)
    with pytest.raises(ValueError, match="non-empty"):
        uniform_logits(torch.empty((0, 2), dtype=torch.long), vocab_size=3)
    with pytest.raises(ValueError, match="positive"):
        uniform_logits(torch.tensor([[0, 1]]), vocab_size=0)


from rebuild_gpt.baselines import unigram_probabilities


def test_unigram_probabilities_use_frequency_and_smoothing():
    probs = unigram_probabilities(torch.tensor([0, 0, 1]), vocab_size=3)
    assert torch.allclose(probs, torch.tensor([3 / 6, 2 / 6, 1 / 6]))
    assert torch.isclose(probs.sum(), torch.tensor(1.0))


def test_unigram_probabilities_reject_invalid_tokens():
    with pytest.raises(ValueError, match="outside"):
        unigram_probabilities(torch.tensor([0, 3]), vocab_size=3)
    with pytest.raises(ValueError, match="outside"):
        unigram_probabilities(torch.tensor([-1, 2]), vocab_size=3)
    with pytest.raises(ValueError, match="non-empty"):
        unigram_probabilities(torch.tensor([], dtype=torch.long), vocab_size=3)
