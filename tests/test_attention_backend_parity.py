"""Both attention implementations must agree on predictions and gradients."""

import pytest
import torch

from rebuild_gpt.attention_backend_parity import compare_attention_backends
from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel


@pytest.mark.parametrize("depth,bias", [(1, True), (2, True), (1, False)])
def test_manual_and_sdpa_attention_match_at_shared_weights(depth, bias):
    model = GPTLanguageModel(GPTConfig(
        11, 5, 8, 2, depth, dropout=0.15,
        attention_impl="fused", bias=bias,
    ))
    model.train()
    before_rng = torch.get_rng_state().clone()
    before_params = [param.detach().clone() for param in model.parameters()]
    x = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    y = torch.tensor([[1, 2, 3, -1], [2, 1, 0, -1]])
    result = compare_attention_backends(model, x, y)
    assert result["matched_parameter_gradients"] > 0
    assert result["max_logit_error"] < 2e-5
    assert result["loss_error"] < 2e-5
    assert result["max_gradient_error"] < 2e-5
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert model.training
    assert all(torch.equal(a, b) for a, b in zip(before_params, model.parameters()))
    assert all(param.grad is None for param in model.parameters())


def test_educational_attention_is_out_of_scope():
    model = GPTLanguageModel(GPTConfig(
        11, 5, 8, 2, 1, attention_impl="educational",
    ))
    with pytest.raises(ValueError, match="fused or sdpa"):
        compare_attention_backends(
            model, torch.tensor([[0, 1]]), torch.tensor([[1, -1]]),
        )


def test_all_ignored_targets_are_rejected():
    model = GPTLanguageModel(GPTConfig(
        11, 5, 8, 2, 1, attention_impl="fused",
    ))
    with pytest.raises(ValueError, match="supervised"):
        compare_attention_backends(
            model, torch.tensor([[0, 1]]), torch.tensor([[-1, -1]]),
        )
