import pytest
import torch

from rebuild_gpt.block import TransformerBlock


def test_block_preserves_shape_and_has_gradients():
    torch.manual_seed(1)
    layer = TransformerBlock(n_embd=8, n_head=2)
    x = torch.randn(2, 4, 8, requires_grad=True)
    output = layer(x)
    assert output.shape == x.shape
    output.square().mean().backward()
    assert x.grad is not None
    assert all(p.grad is not None for p in layer.parameters())


def test_block_does_not_read_future_token_vectors():
    torch.manual_seed(3)
    block = TransformerBlock(8, 2)
    block.eval()
    x = torch.randn(2, 5, 8)
    changed = x.clone()
    changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 25
    with torch.no_grad():
        assert torch.allclose(block(x)[:, :3], block(changed)[:, :3], atol=1e-5)
        assert torch.allclose(block(x)[:, :3], block(x[:, :3]), atol=1e-5)


def test_zeroed_transformations_leave_only_residual_signal():
    layer = TransformerBlock(6, 2)
    with torch.no_grad():
        for name, param in layer.named_parameters():
            if name.startswith("attention.") or name.startswith("mlp."):
                param.zero_()
    x = torch.randn(2, 3, 6)
    assert torch.allclose(layer(x), x)


def test_block_rejects_incompatible_head_configuration():
    with pytest.raises(ValueError, match="divisible"):
        TransformerBlock(7, 2)
