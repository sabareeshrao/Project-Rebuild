import pytest
import torch

from rebuild_gpt.position_model import EmbeddingLanguageModel


def test_position_model_maps_batch_time_to_vocab_logits():
    model = EmbeddingLanguageModel(vocab_size=5, block_size=4, n_embd=3)
    logits = model(torch.tensor([[1, 2, 3], [3, 2, 1]]))
    assert logits.shape == (2, 3, 5)
    assert sum(p.numel() for p in model.parameters()) == 5 * 3 + 4 * 3 + 3 * 5


def test_position_model_responds_to_position_not_earlier_context():
    model = EmbeddingLanguageModel(vocab_size=4, block_size=3, n_embd=2)
    with torch.no_grad():
        model.positions.lookup.weight.copy_(
            torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        )
        model.head.linear.weight.fill_(1.0)
    logits = model(torch.tensor([[0, 1, 2], [3, 1, 2]]))
    assert torch.equal(logits[0, 1], logits[1, 1])
    assert torch.equal(logits[0, 2], logits[1, 2])
    assert not torch.equal(logits[0, 1], logits[0, 2])


def test_position_model_rejects_context_longer_than_block_size():
    model = EmbeddingLanguageModel(vocab_size=3, block_size=2, n_embd=4)
    with pytest.raises(ValueError, match="block_size"):
        model(torch.tensor([[0, 1, 2]]))


def test_position_model_loss_backpropagates_through_all_three_tables():
    model = EmbeddingLanguageModel(3, 4, 5)
    loss = model.loss(torch.tensor([[0, 1, 2]]), torch.tensor([[1, 2, 0]]))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


from rebuild_gpt.training import train_step


def test_position_model_learns_a_position_dependent_toy_pattern():
    torch.manual_seed(4)
    model = EmbeddingLanguageModel(vocab_size=2, block_size=4, n_embd=8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    x = torch.tensor([[0, 0, 0, 0]] * 8)
    y = torch.tensor([[0, 1, 0, 1]] * 8)
    before = model.loss(x, y).item()
    for _ in range(70):
        train_step(model, optimizer, x, y)
    after = model.loss(x, y).item()
    assert after < before
    assert after < 0.15
