"""Partial recomputation must preserve loss, dropout, gradients and updates."""

from dataclasses import replace

import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.training import train_step


def matching_pair(backend: str):
    config = GPTConfig(
        vocab_size=9, block_size=4, n_embd=8, n_head=2, n_layer=4,
        dropout=0.2, attention_impl=backend,
    )
    torch.manual_seed(128)
    normal = GPTLanguageModel(config)
    selected = GPTLanguageModel(replace(config))
    selected.load_state_dict(normal.state_dict(), strict=True)
    return normal, selected


@pytest.mark.parametrize("backend", ["fused", "sdpa"])
@pytest.mark.parametrize("every_n", [1, 2, 3, 4])
def test_checkpoint_interval_preserves_dropout_loss_gradients_and_rng(backend, every_n):
    normal, selected = matching_pair(backend)
    selected.blocks.set_checkpoint_interval(every_n)
    normal.train()
    selected.train()
    inputs = torch.tensor([[0, 1, 2], [3, 4, 5]])
    targets = torch.tensor([[1, 2, 3], [4, 5, 6]])

    torch.manual_seed(235)
    baseline_loss = normal.loss(inputs, targets)
    baseline_loss.backward()
    baseline_rng = torch.get_rng_state().clone()

    torch.manual_seed(235)
    selective_loss = selected.loss(inputs, targets)
    selective_loss.backward()
    selective_rng = torch.get_rng_state().clone()

    torch.testing.assert_close(
        selective_loss, baseline_loss, atol=2e-5, rtol=2e-5
    )
    for (a_name, a), (b_name, b) in zip(
        normal.named_parameters(), selected.named_parameters()
    ):
        assert a_name == b_name
        assert a.grad is not None and b.grad is not None
        torch.testing.assert_close(a.grad, b.grad, atol=2e-5, rtol=2e-5)
    assert torch.equal(baseline_rng, selective_rng)


def test_optimizer_update_is_equivalent_with_every_second_block():
    normal, selected = matching_pair("sdpa")
    selected.blocks.set_checkpoint_interval(2)
    opt_a = torch.optim.AdamW(normal.parameters(), lr=0.001)
    opt_b = torch.optim.AdamW(selected.parameters(), lr=0.001)
    inputs = torch.tensor([[0, 1, 2], [3, 4, 5]])
    targets = torch.tensor([[1, 2, 3], [4, 5, 6]])
    torch.manual_seed(456)
    loss_a = train_step(normal, opt_a, inputs, targets)
    torch.manual_seed(456)
    loss_b = train_step(selected, opt_b, inputs, targets)
    assert loss_b == pytest.approx(loss_a, abs=2e-5)
    for a, b in zip(normal.parameters(), selected.parameters()):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)


def test_selective_checkpointing_keeps_checkpoint_tensor_names():
    normal, selected = matching_pair("fused")
    selected.blocks.set_checkpoint_interval(2)
    assert list(normal.state_dict()) == list(selected.state_dict())
    assert all(
        torch.equal(normal.state_dict()[name], selected.state_dict()[name])
        for name in normal.state_dict()
    )
