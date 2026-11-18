"""Opt-in tests against the genuine upstream nanoGPT model.py."""

import os
from pathlib import Path

import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.upstream_parity import compare_against_upstream, load_upstream_model


_upstream_path = os.environ.get("NANOGPT_MODEL_PY", "")
pytestmark = pytest.mark.skipif(
    not _upstream_path or not Path(_upstream_path).is_file(),
    reason="Set NANOGPT_MODEL_PY to a trusted original nanoGPT model.py for parity tests",
)


@pytest.mark.parametrize("bias,depth", [(True, 1), (True, 2), (False, 1)])
def test_actual_original_nanogpt_matches_our_fused_gpt(bias, depth):
    torch.manual_seed(17)
    cfg = GPTConfig(
        vocab_size=17,
        block_size=6,
        n_embd=12,
        n_head=3,
        n_layer=depth,
        dropout=0.1,
        attention_impl="fused",
        bias=bias,
    )
    ours = GPTLanguageModel(cfg)
    upstream = load_upstream_model(_upstream_path, cfg)
    inputs = torch.tensor([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
    targets = torch.tensor([[1, 2, 3, 4, -1], [3, 2, 1, -1, 5]])
    result = compare_against_upstream(ours, upstream, inputs, targets)
    assert result.matched_tensors > 0
    assert result.max_logit_error < 2e-5
    assert result.loss_error < 2e-5
