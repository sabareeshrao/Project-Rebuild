"""Opt-in token-for-token comparisons against original nanoGPT's generator."""

import os
from pathlib import Path

import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.generation_parity import compare_generation_against_upstream
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.upstream_parity import load_upstream_model


_reference = os.environ.get("NANOGPT_MODEL_PY", "")
pytestmark = pytest.mark.skipif(
    not _reference or not Path(_reference).is_file(),
    reason="Set NANOGPT_MODEL_PY to a trusted original nanoGPT model.py",
)


@pytest.mark.parametrize("temperature,top_k", [(1.0, None), (0.75, 3), (1.5, 1)])
def test_original_nanogpt_generation_matches_after_weight_transfer(temperature, top_k):
    torch.manual_seed(17)
    config = GPTConfig(
        vocab_size=17, block_size=5, n_embd=12, n_head=3, n_layer=2,
        attention_impl="fused", bias=True, dropout=0.0,
    )
    ours = GPTLanguageModel(config)
    upstream = load_upstream_model(_reference, config)
    # Prompt length exceeds block_size to exercise the cropping behavior.
    prompt = torch.tensor([[0, 1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1, 0]])
    report = compare_generation_against_upstream(
        ours, upstream, prompt, 8, temperature=temperature, top_k=top_k, seed=33,
    )
    assert report.matched_tensors > 0
    assert report.new_tokens == 8
