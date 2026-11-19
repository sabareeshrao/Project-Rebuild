"""Changing a future token cannot change earlier GPT predictions."""

import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.model_invariants import check_causal_prefix


@pytest.mark.parametrize("implementation", ["fused", "sdpa"])
def test_causal_prefix_is_unchanged_for_both_attention_paths(implementation):
    model = GPTLanguageModel(GPTConfig(
        11, 6, 12, 3, 2, dropout=0.2, attention_impl=implementation,
    ))
    model.train()
    inputs = torch.tensor([[1, 2, 3, 4], [6, 5, 4, 3]])
    report = check_causal_prefix(model, inputs)
    assert report["checked_positions"] == 6
    assert report["max_prefix_error"] <= 2e-5
    assert model.training


class LeakyModel(GPTLanguageModel):
    def forward(self, tokens):
        output = super().forward(tokens)
        # Deliberately leak the final token into every earlier prediction.
        return output + tokens[:, -1:].unsqueeze(-1).float()


def test_counterexample_catches_a_future_token_leak():
    model = LeakyModel(GPTConfig(11, 5, 8, 2, 1, attention_impl="fused"))
    model.eval()
    with pytest.raises(AssertionError, match="future token"):
        check_causal_prefix(model, torch.tensor([[1, 2, 3, 4]]))
    assert not model.training


@pytest.mark.parametrize("bad", [
    torch.tensor([1, 2]), torch.tensor([[1]]),
    torch.tensor([[1., 2.]]), torch.tensor([[99, 1]]),
])
def test_invalid_input_rejected_before_forward(bad):
    model = GPTLanguageModel(GPTConfig(11, 5, 8, 2, 1, attention_impl="fused"))
    with pytest.raises(ValueError):
        check_causal_prefix(model, bad)
