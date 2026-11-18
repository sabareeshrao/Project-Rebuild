import copy

import pytest
import torch
from torch import nn

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.generation_parity import compare_generation_against_upstream
from rebuild_gpt.gpt import GPTLanguageModel


class LocalProxy(nn.Module):
    """A local test double, not a substitute for the actual upstream test."""

    def __init__(self, model):
        super().__init__()
        self.model = copy.deepcopy(model)

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        return self.model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)


def test_parity_helper_preserves_rng_and_module_modes(monkeypatch):
    torch.manual_seed(10)
    model = GPTLanguageModel(GPTConfig(7, 4, 8, 2, 1, dropout=0.2, attention_impl="fused"))
    reference = LocalProxy(model)
    model.train()
    reference.train()
    monkeypatch.setattr(
        "rebuild_gpt.generation_parity.synchronize_upstream_weights",
        lambda ours, upstream: 42,
    )
    before = torch.get_rng_state().clone()
    prompt = torch.tensor([[1, 2, 3, 4, 5]])
    result = compare_generation_against_upstream(
        model, reference, prompt, 5, temperature=0.7, top_k=3, seed=17
    )
    assert result.matched_tensors == 42
    assert result.new_tokens == 5
    assert torch.equal(before, torch.get_rng_state())
    assert model.training and reference.training


def test_parity_helper_detects_generated_token_mismatch(monkeypatch):
    torch.manual_seed(8)
    model = GPTLanguageModel(GPTConfig(7, 4, 8, 2, 1, attention_impl="fused"))

    class WrongProxy(LocalProxy):
        def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
            result = super().generate(idx, max_new_tokens, temperature, top_k).clone()
            result[:, -1] = (result[:, -1] + 1) % 7
            return result

    monkeypatch.setattr(
        "rebuild_gpt.generation_parity.synchronize_upstream_weights",
        lambda ours, upstream: 1,
    )
    with pytest.raises(AssertionError, match="Generation tokens differ"):
        compare_generation_against_upstream(
            model, WrongProxy(model), torch.tensor([[1, 2]]), 3
        )


def test_parity_rejects_bad_seed_before_using_reference(monkeypatch):
    model = GPTLanguageModel(GPTConfig(7, 4, 8, 2, 1, attention_impl="fused"))
    with pytest.raises(ValueError, match="seed"):
        compare_generation_against_upstream(model, LocalProxy(model), torch.tensor([[1]]), 1, seed=-1)
