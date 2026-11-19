"""A decoder attention mask must isolate batch rows as well as future tokens."""

import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.model_invariants import check_batch_independence


@pytest.mark.parametrize("impl", ["fused", "sdpa"])
def test_full_batch_matches_separate_forward_calls(impl):
    model = GPTLanguageModel(GPTConfig(
        13, 6, 12, 3, 2, dropout=0.2, attention_impl=impl,
    ))
    model.train()
    inputs = torch.tensor([[0, 1, 2, 3], [9, 8, 7, 6], [2, 4, 6, 8]])
    report = check_batch_independence(model, inputs)
    assert report["checked_samples"] == 3
    assert report["max_batch_error"] < 2e-5
    assert model.training


class BatchLeakingModel(GPTLanguageModel):
    def forward(self, tokens):
        logits = super().forward(tokens)
        # This is deliberately WRONG: output depends on other batch rows.
        return logits + tokens.float().sum().reshape(1, 1, 1)


def test_cross_example_dependency_is_detected():
    model = BatchLeakingModel(GPTConfig(11, 5, 8, 2, 1, attention_impl="fused"))
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(AssertionError, match="Batch independence"):
        check_batch_independence(model, tokens)


def test_one_row_is_not_an_independence_experiment():
    model = GPTLanguageModel(GPTConfig(11, 5, 8, 2, 1, attention_impl="fused"))
    with pytest.raises(ValueError, match="batch"):
        check_batch_independence(model, torch.tensor([[1, 2, 3]]))
