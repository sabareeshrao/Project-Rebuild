"""Checkpointed transformer blocks recompute forward during backward."""

import pytest
import torch

from rebuild_gpt.stack import TransformerStack


def make_pair():
    torch.manual_seed(122)
    regular = TransformerStack(8, 2, 2, attention_impl="sdpa")
    checkpointed = TransformerStack(8, 2, 2, attention_impl="sdpa")
    checkpointed.load_state_dict(regular.state_dict(), strict=True)
    checkpointed.set_activation_checkpointing(True)
    return regular, checkpointed


def test_checkpointing_preserves_output_and_gradients():
    regular, checkpointed = make_pair()
    x = torch.randn(2, 4, 8, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    output_a = regular(x)
    output_b = checkpointed(y)
    torch.testing.assert_close(output_a, output_b, atol=1e-6, rtol=1e-6)
    output_a.square().mean().backward()
    output_b.square().mean().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=1e-6, rtol=1e-6)
    for a, b in zip(regular.parameters(), checkpointed.parameters()):
        assert a.grad is not None and b.grad is not None
        torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-6)


def test_activation_checkpointing_does_not_add_state_dict_tensors():
    regular, checkpointed = make_pair()
    assert list(regular.state_dict()) == list(checkpointed.state_dict())


def test_flag_rejects_nonboolean_values():
    with pytest.raises(ValueError, match="boolean"):
        TransformerStack(8, 2, 1, activation_checkpointing=1)
    stack = TransformerStack(8, 2, 1)
    with pytest.raises(ValueError, match="boolean"):
        stack.set_activation_checkpointing("true")


def test_disabled_or_eval_mode_avoids_recomputation(monkeypatch):
    import rebuild_gpt.stack as stack_module

    regular, checkpointed = make_pair()
    calls = []
    def fail_if_called(*args, **kwargs):
        calls.append(True)
        raise AssertionError("checkpoint should not run outside grad-enabled training")

    monkeypatch.setattr(stack_module, "checkpoint", fail_if_called)
    x = torch.randn(2, 3, 8, requires_grad=True)
    checkpointed.eval()
    checkpointed(x)
    checkpointed.train()
    with torch.no_grad():
        checkpointed(x)
    regular(x)
    assert not calls
