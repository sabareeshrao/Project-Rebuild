import pytest
import torch

from rebuild_gpt.multihead_model import MultiHeadLanguageModel
from rebuild_gpt.training import train_step


def test_multihead_model_produces_logits_and_all_parameter_gradients():
    torch.manual_seed(2)
    model = MultiHeadLanguageModel(vocab_size=5, block_size=4, n_embd=8, n_head=2)
    x = torch.tensor([[0, 1, 2, 3], [4, 3, 2, 1]])
    y = torch.tensor([[1, 2, 3, 4], [3, 2, 1, 0]])
    logits = model(x)
    assert logits.shape == (2, 4, 5)
    model.loss(x, y).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_multihead_model_preserves_causality_across_prefix():
    torch.manual_seed(4)
    model = MultiHeadLanguageModel(6, 6, 12, 3, dropout=0.0)
    model.eval()
    base = torch.tensor([[0, 1, 2, 3, 4]])
    changed = torch.tensor([[0, 1, 2, 5, 5]])
    with torch.no_grad():
        assert torch.allclose(model(base)[:, :3], model(changed)[:, :3], atol=1e-6)
        assert torch.allclose(model(base)[:, :3], model(base[:, :3]), atol=1e-6)


def test_multihead_model_can_use_generic_training_step():
    torch.manual_seed(0)
    model = MultiHeadLanguageModel(3, 2, 8, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    x = torch.tensor([[0, 1], [2, 1]])
    y = torch.tensor([[1, 0], [1, 2]])
    before = model.tokens.lookup.weight.detach().clone()
    loss = train_step(model, optimizer, x, y)
    assert loss > 0
    assert not torch.equal(before, model.tokens.lookup.weight)


def test_multihead_model_checks_block_size():
    model = MultiHeadLanguageModel(3, 2, 8, 2)
    with pytest.raises(ValueError, match="block_size"):
        model(torch.tensor([[0, 1, 2]]))
