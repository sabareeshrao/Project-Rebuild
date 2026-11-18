import pytest
import torch

from rebuild_gpt.training import train_step
from rebuild_gpt.transformer_model import TransformerLanguageModel


def test_transformer_language_model_shapes_and_all_gradients():
    torch.manual_seed(2)
    model = TransformerLanguageModel(vocab_size=5, block_size=4, n_embd=8, n_head=2)
    x = torch.tensor([[0, 1, 2, 3], [4, 3, 2, 1]])
    y = torch.tensor([[1, 2, 3, 4], [3, 2, 1, 0]])
    assert model(x).shape == (2, 4, 5)
    model.loss(x, y).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_transformer_language_model_is_causal():
    torch.manual_seed(4)
    model = TransformerLanguageModel(6, 6, 12, 3, dropout=0.0)
    model.eval()
    first = torch.tensor([[0, 1, 2, 3, 4]])
    future_changed = torch.tensor([[0, 1, 2, 5, 5]])
    with torch.no_grad():
        assert torch.allclose(model(first)[:, :3], model(future_changed)[:, :3], atol=1e-5)
        assert torch.allclose(model(first)[:, :3], model(first[:, :3]), atol=1e-5)


def test_one_block_model_can_learn_from_earlier_context():
    torch.manual_seed(7)
    model = TransformerLanguageModel(vocab_size=3, block_size=2, n_embd=8, n_head=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    x = torch.tensor([[0, 1]] * 8 + [[2, 1]] * 8)
    y = torch.tensor([[1, 0]] * 8 + [[1, 2]] * 8)
    before = model.loss(x, y).item()
    for _ in range(110):
        train_step(model, optimizer, x, y)
    model.eval()
    with torch.no_grad():
        after = model.loss(x, y).item()
        guesses = model(x).argmax(-1)
    assert after < before
    assert after < 0.1
    assert torch.equal(guesses, y)


def test_one_block_model_rejects_excessive_context():
    model = TransformerLanguageModel(3, 2, 8, 2)
    with pytest.raises(ValueError, match="block_size"):
        model(torch.tensor([[0, 1, 2]]))
