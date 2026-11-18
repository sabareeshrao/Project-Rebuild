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
