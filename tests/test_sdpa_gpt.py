"""SDPA is a runtime attention choice, not a different set of weights."""

from dataclasses import replace

import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.fused_attention import FusedCausalSelfAttention
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.nanogpt_keys import export_nanogpt_state_dict
from rebuild_gpt.training import train_step


def models():
    config = GPTConfig(11, 6, 12, 3, 2, attention_impl="fused")
    torch.manual_seed(73)
    manual = GPTLanguageModel(config)
    sdpa = GPTLanguageModel(replace(config, attention_impl="sdpa"))
    sdpa.load_state_dict(manual.state_dict(), strict=True)
    return manual, sdpa


def test_sdpa_is_selected_in_every_block_without_changing_keys():
    manual, sdpa = models()
    assert all(
        isinstance(block.attention, FusedCausalSelfAttention)
        and block.attention.use_sdpa
        for block in sdpa.blocks.blocks
    )
    assert list(sdpa.state_dict()) == list(manual.state_dict())


def test_full_gpt_logits_match_with_identical_weights():
    manual, sdpa = models()
    manual.eval()
    sdpa.eval()
    ids = torch.tensor([[0, 1, 2, 3, 4], [2, 1, 3, 0, 4]])
    with torch.no_grad():
        torch.testing.assert_close(
            manual(ids), sdpa(ids), atol=2e-5, rtol=2e-5
        )


def test_sdpa_supports_bias_free_upstream_key_translation():
    config = GPTConfig(7, 4, 8, 2, 1, attention_impl="sdpa", bias=False)
    model = GPTLanguageModel(config)
    exported = export_nanogpt_state_dict(model)
    assert exported
    assert all(not name.endswith(".bias") for name in exported)


def test_sdpa_gpt_can_take_a_training_step():
    _, model = models()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    x = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    y = torch.tensor([[1, 2, 3, 4], [2, 1, 0, 5]])
    loss = train_step(model, optimizer, x, y)
    assert torch.isfinite(torch.tensor(loss))
    assert all(p.grad is not None for p in model.parameters())
