import pytest
import torch

from rebuild_gpt.stack import TransformerStack


def test_stack_preserves_shapes_and_has_independent_blocks():
    torch.manual_seed(2)
    stack = TransformerStack(n_embd=8, n_head=2, n_layer=3)
    hidden = torch.randn(2, 4, 8, requires_grad=True)
    assert len(stack.blocks) == 3
    assert stack.blocks[0] is not stack.blocks[1]
    output = stack(hidden)
    assert output.shape == hidden.shape
    output.square().sum().backward()
    assert hidden.grad is not None
    assert all(parameter.grad is not None for parameter in stack.parameters())


def test_stack_rejects_zero_layers():
    with pytest.raises(ValueError, match="n_layer"):
        TransformerStack(8, 2, 0)


def test_stack_keeps_prefix_independent_of_future():
    torch.manual_seed(3)
    stack = TransformerStack(8, 2, 3)
    stack.eval()
    x = torch.randn(2, 5, 8)
    changed = x.clone()
    changed[:, 3:] += 50
    with torch.no_grad():
        assert torch.allclose(stack(x)[:, :3], stack(changed)[:, :3], atol=1e-5)
        assert torch.allclose(stack(x)[:, :3], stack(x[:, :3]), atol=1e-5)
