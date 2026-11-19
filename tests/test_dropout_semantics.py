"""Dropout must be deterministic during evaluation, stochastic during training."""

import pytest
import torch
from torch import nn

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.dropout_semantics import check_dropout_semantics
from rebuild_gpt.gpt import GPTLanguageModel


def _model(*, dropout=0.4):
    return GPTLanguageModel(GPTConfig(
        11, 5, 12, 3, 2, dropout=dropout, attention_impl="fused",
    ))


def _tokens():
    return torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])


def test_dropout_repeatability_and_rng_restoration():
    model = _model()
    model.eval()
    rng_before = torch.get_rng_state().clone()
    report = check_dropout_semantics(model, _tokens(), seed=29)
    assert report["status"] == "passed"
    assert report["eval_repeat_error"] == 0.0
    assert report["training_output_difference"] > 0.0
    assert not model.training
    assert torch.equal(rng_before, torch.get_rng_state())


def test_disabled_dropout_cannot_masquerade_as_working_dropout():
    model = _model()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    with pytest.raises(AssertionError, match="identical predictions"):
        check_dropout_semantics(model, _tokens())


def test_zero_dropout_is_not_a_stochastic_experiment():
    with pytest.raises(ValueError, match="between zero and one"):
        check_dropout_semantics(_model(dropout=0.0), _tokens())


@pytest.mark.parametrize("seed", [-1, True, 1.1])
def test_invalid_seed_is_rejected(seed):
    with pytest.raises(ValueError, match="seed"):
        check_dropout_semantics(_model(), _tokens(), seed=seed)


def test_dropout_failure_restores_original_mode_and_rng():
    model = _model()
    model.train()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    rng_before = torch.get_rng_state().clone()
    with pytest.raises(AssertionError):
        check_dropout_semantics(model, _tokens())
    assert model.training
    assert torch.equal(torch.get_rng_state(), rng_before)
