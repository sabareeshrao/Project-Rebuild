import pytest
import torch
from torch import nn

from rebuild_gpt.embeddings import TokenEmbedding


def test_tokens_become_vectors_without_losing_batch_time():
    model = TokenEmbedding(vocab_size=5, n_embd=3)
    tokens = torch.tensor([[0, 1, 2], [4, 1, 3]])
    vectors = model(tokens)
    assert vectors.shape == (2, 3, 3)
    assert torch.equal(vectors[0, 1], vectors[1, 1])
    assert model.lookup.weight.shape == (5, 3)


def test_token_embedding_gradients_reach_lookup_table():
    model = TokenEmbedding(4, 6)
    model(torch.tensor([[1, 2]])).sum().backward()
    grad = model.lookup.weight.grad
    assert grad is not None
    assert grad[1].abs().sum() > 0 and grad[2].abs().sum() > 0
    assert torch.equal(grad[0], torch.zeros_like(grad[0]))


@pytest.mark.parametrize("shape", [(3,), (0, 4)])
def test_token_embeddings_reject_wrong_batch_shape(shape):
    model = TokenEmbedding(5, 2)
    with pytest.raises(ValueError, match="non-empty"):
        model(torch.zeros(shape, dtype=torch.long))


def test_token_embeddings_reject_out_of_range_ids():
    model = TokenEmbedding(5, 2)
    with pytest.raises(ValueError, match="outside"):
        model(torch.tensor([[5]]))
