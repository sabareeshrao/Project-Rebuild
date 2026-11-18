import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel


def test_gpt_model_shapes_depth_and_gradients():
    torch.manual_seed(4)
    model = GPTLanguageModel(GPTConfig(vocab_size=5, block_size=4, n_embd=8, n_head=2, n_layer=3))
    x = torch.tensor([[0, 1, 2, 3], [4, 3, 2, 1]])
    y = torch.tensor([[1, 2, 3, 4], [3, 2, 1, 0]])
    assert len(model.blocks.blocks) == 3
    assert model(x).shape == (2, 4, 5)
    model.loss(x, y).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_gpt_rejects_more_tokens_than_context_window():
    model = GPTLanguageModel(GPTConfig(5, 3, 8, 2, 2))
    with pytest.raises(ValueError, match="block_size"):
        model(torch.tensor([[0, 1, 2, 3]]))


def test_multiple_layers_preserve_autoregressive_causality():
    torch.manual_seed(10)
    model = GPTLanguageModel(GPTConfig(6, 6, 12, 3, 3))
    model.eval()
    prefix = torch.tensor([[0, 1, 2]])
    first = torch.tensor([[0, 1, 2, 3, 4]])
    future_changed = torch.tensor([[0, 1, 2, 5, 5]])
    with torch.no_grad():
        assert torch.allclose(model(first)[:, :3], model(future_changed)[:, :3], atol=1e-5)
        assert torch.allclose(model(first)[:, :3], model(prefix), atol=1e-5)
