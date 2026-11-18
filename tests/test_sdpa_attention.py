"""The optimized attention kernel must preserve causal semantics."""

import pytest
import torch

from rebuild_gpt.fused_attention import FusedCausalSelfAttention


@pytest.mark.parametrize("time", [1, 2, 7])
def test_sdpa_matches_manual_forward_when_dropout_disabled(time):
    torch.manual_seed(91 + time)
    manual = FusedCausalSelfAttention(12, 3, dropout=0.0, use_sdpa=False).eval()
    optimized = FusedCausalSelfAttention(12, 3, dropout=0.0, use_sdpa=True).eval()
    optimized.load_state_dict(manual.state_dict(), strict=True)
    hidden = torch.randn(2, time, 12)
    with torch.no_grad():
        expected = manual(hidden)
        actual = optimized(hidden)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_sdpa_does_not_change_state_dict_keys_or_params():
    manual = FusedCausalSelfAttention(8, 2)
    sdpa = FusedCausalSelfAttention(8, 2, use_sdpa=True)
    assert list(manual.state_dict()) == list(sdpa.state_dict())


def test_sdpa_rejects_invalid_flag():
    with pytest.raises(ValueError, match="use_sdpa"):
        FusedCausalSelfAttention(8, 2, use_sdpa=1)


@pytest.mark.parametrize("time", [1, 2, 7])
def test_sdpa_matches_manual_backpropagation(time):
    torch.manual_seed(180 + time)
    manual = FusedCausalSelfAttention(12, 3, use_sdpa=False).train()
    sdpa = FusedCausalSelfAttention(12, 3, use_sdpa=True).train()
    sdpa.load_state_dict(manual.state_dict(), strict=True)
    a = torch.randn(2, time, 12, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    manual(a).square().mean().backward()
    sdpa(b).square().mean().backward()
    torch.testing.assert_close(a.grad, b.grad, atol=1e-5, rtol=1e-5)
    for x, y in zip(manual.parameters(), sdpa.parameters()):
        torch.testing.assert_close(x.grad, y.grad, atol=1e-5, rtol=1e-5)


def test_sdpa_preserves_prefix_when_future_tokens_change():
    torch.manual_seed(24)
    sdpa = FusedCausalSelfAttention(12, 3, use_sdpa=True).eval()
    tokens = torch.randn(2, 7, 12)
    altered = tokens.clone()
    altered[:, 3:, :] *= -40
    with torch.no_grad():
        torch.testing.assert_close(
            sdpa(tokens)[:, :3, :], sdpa(altered)[:, :3, :],
            atol=1e-5, rtol=1e-5,
        )
        torch.testing.assert_close(
            sdpa(tokens)[:, :3, :], sdpa(tokens[:, :3])[:, :3, :],
            atol=1e-5, rtol=1e-5,
        )


def test_sdpa_eval_mode_disables_attention_dropout():
    torch.manual_seed(8)
    layer = FusedCausalSelfAttention(8, 2, dropout=0.75, use_sdpa=True).eval()
    tokens = torch.randn(2, 6, 8)
    with torch.no_grad():
        torch.testing.assert_close(layer(tokens), layer(tokens), atol=0, rtol=0)


def test_sdpa_train_mode_keeps_attention_dropout():
    torch.manual_seed(11)
    layer = FusedCausalSelfAttention(8, 2, dropout=0.9, use_sdpa=True).train()
    tokens = torch.randn(4, 6, 8)
    assert not torch.equal(layer(tokens), layer(tokens))
