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
