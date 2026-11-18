"""Compare complete GPT gradients and optimizer results across attention kernels."""

from dataclasses import replace

import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.training import train_step


def matching_models():
    torch.manual_seed(112)
    cfg = GPTConfig(
        vocab_size=11, block_size=6, n_embd=12, n_head=3,
        n_layer=2, dropout=0.0, attention_impl="fused",
    )
    manual = GPTLanguageModel(cfg)
    sdpa = GPTLanguageModel(replace(cfg, attention_impl="sdpa"))
    sdpa.load_state_dict(manual.state_dict(), strict=True)
    return manual, sdpa


@pytest.mark.parametrize("length", [1, 2, 5])
def test_gpt_gradients_match_for_different_context_lengths(length):
    manual, sdpa = matching_models()
    tokens = torch.arange(length).unsqueeze(0).repeat(2, 1)
    targets = (tokens + 1) % manual.config.vocab_size
    first = manual.loss(tokens, targets)
    second = sdpa.loss(tokens, targets)
    torch.testing.assert_close(first, second, rtol=2e-5, atol=2e-5)
    first.backward()
    second.backward()
    for (name_a, a), (name_b, b) in zip(
        manual.named_parameters(), sdpa.named_parameters()
    ):
        assert name_a == name_b
        assert a.grad is not None and b.grad is not None
        torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=2e-5)


def test_adamw_update_matches_after_one_training_step():
    manual, sdpa = matching_models()
    opt_manual = torch.optim.AdamW(manual.parameters(), lr=0.001)
    opt_sdpa = torch.optim.AdamW(sdpa.parameters(), lr=0.001)
    x = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    y = torch.tensor([[1, 2, 3, 4], [2, 1, 0, 5]])
    loss_a = train_step(manual, opt_manual, x, y)
    loss_b = train_step(sdpa, opt_sdpa, x, y)
    assert loss_a == pytest.approx(loss_b, abs=2e-5)
    for (name_a, a), (name_b, b) in zip(
        manual.named_parameters(), sdpa.named_parameters()
    ):
        assert name_a == name_b
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)
