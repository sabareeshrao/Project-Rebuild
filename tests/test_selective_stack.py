"""The stack wraps only the selected blocks during training."""

import pytest
import torch

from rebuild_gpt.stack import TransformerStack


def make_stack():
    return TransformerStack(
        n_embd=8, n_head=2, n_layer=4,
        attention_impl="sdpa", dropout=0.0,
    )


def test_legacy_all_or_none_controls_are_preserved():
    stack = make_stack()
    assert stack.checkpointed_blocks == ()
    stack.set_activation_checkpointing(True)
    assert stack.activation_checkpointing
    assert stack.checkpointed_blocks == (0, 1, 2, 3)
    stack.set_activation_checkpointing(False)
    assert stack.checkpointed_blocks == ()


def test_every_second_checkpoint_wraps_only_two_blocks(monkeypatch):
    import rebuild_gpt.stack as stack_module

    stack = make_stack().train()
    stack.set_checkpoint_interval(2)
    assert stack.checkpointed_blocks == (1, 3)
    assert not stack.activation_checkpointing
    indices = []
    def spy(block, tensor, **kwargs):
        indices.append(next(i for i, item in enumerate(stack.blocks) if item is block))
        assert kwargs["use_reentrant"] is False
        assert kwargs["preserve_rng_state"] is True
        return block(tensor)

    monkeypatch.setattr(stack_module, "checkpoint", spy)
    assert stack(torch.randn(2, 3, 8, requires_grad=True)).shape == (2, 3, 8)
    assert indices == [1, 3]


def test_eval_and_no_grad_skip_selected_recomputation(monkeypatch):
    import rebuild_gpt.stack as stack_module

    stack = make_stack()
    stack.set_checkpoint_interval(2)
    def unexpected(*args, **kwargs):
        raise AssertionError("checkpointing must be training-only")
    monkeypatch.setattr(stack_module, "checkpoint", unexpected)
    x = torch.randn(2, 3, 8)
    stack.eval()
    stack(x)
    stack.train()
    with torch.no_grad():
        stack(x)


def test_selective_policy_can_be_reset_and_legacy_mode_overrides_it():
    stack = make_stack()
    stack.set_checkpoint_interval(3)
    assert stack.checkpointed_blocks == (2,)
    stack.set_activation_checkpointing(True)
    assert stack.checkpointed_blocks == (0, 1, 2, 3)
    stack.set_checkpoint_interval(None)
    assert stack.checkpointed_blocks == ()


def test_invalid_interval_does_not_change_prior_policy():
    stack = make_stack()
    stack.set_checkpoint_interval(2)
    with pytest.raises(ValueError, match="every_n"):
        stack.set_checkpoint_interval(5)
    assert stack.checkpointed_blocks == (1, 3)
