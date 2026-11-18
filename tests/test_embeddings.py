import pytest
import torch
from torch import nn

from rebuild_gpt.embeddings import TokenEmbedding, VocabularyProjection, PositionEmbedding


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


def test_vocab_projection_produces_logits_with_expected_shape():
    head = VocabularyProjection(n_embd=3, vocab_size=5)
    vectors = torch.randn(2, 4, 3, requires_grad=True)
    logits = head(vectors)
    assert logits.shape == (2, 4, 5)
    assert isinstance(head.linear, nn.Linear)
    assert head.linear.bias is None
    logits.sum().backward()
    assert vectors.grad is not None
    assert head.linear.weight.grad is not None


def test_vocab_projection_rejects_wrong_feature_count():
    head = VocabularyProjection(3, 5)
    with pytest.raises(ValueError, match="n_embd"):
        head(torch.zeros(2, 4, 2))


def test_position_embedding_gives_each_index_its_own_vector():
    position = PositionEmbedding(block_size=6, n_embd=3)
    with torch.no_grad():
        position.lookup.weight.copy_(torch.arange(18).reshape(6, 3).float())
    vectors = position(3)
    assert vectors.shape == (3, 3)
    assert torch.equal(vectors[0], torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(vectors[2], torch.tensor([6.0, 7.0, 8.0]))


@pytest.mark.parametrize("length", [0, 5, -1])
def test_position_embedding_rejects_invalid_context_length(length):
    with pytest.raises(ValueError, match="block_size"):
        PositionEmbedding(4, 3)(length)
