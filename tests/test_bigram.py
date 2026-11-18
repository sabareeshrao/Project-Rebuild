import pytest
import torch
from torch import nn

from rebuild_gpt.bigram import BigramLanguageModel


def test_bigram_logits_have_batch_time_vocab_shape():
    model = BigramLanguageModel(vocab_size=4)
    ids = torch.tensor([[0, 1, 2], [3, 2, 1]])
    logits = model(ids)
    assert logits.shape == (2, 3, 4)
    assert isinstance(model.token_to_logits, nn.Embedding)
    assert sum(parameter.numel() for parameter in model.parameters()) == 16


def test_same_current_token_has_identical_logits_regardless_of_context():
    model = BigramLanguageModel(vocab_size=4)
    logits = model(torch.tensor([[0, 2], [1, 2]]))
    assert torch.equal(logits[0, 1], logits[1, 1])


@pytest.mark.parametrize("vocab_size", [0, 1])
def test_bigram_requires_at_least_two_tokens(vocab_size):
    with pytest.raises(ValueError, match="at least 2"):
        BigramLanguageModel(vocab_size)


def test_bigram_rejects_invalid_token_ids():
    model = BigramLanguageModel(vocab_size=3)
    with pytest.raises(ValueError, match="long tensor"):
        model(torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="outside vocabulary"):
        model(torch.tensor([[3]]))
