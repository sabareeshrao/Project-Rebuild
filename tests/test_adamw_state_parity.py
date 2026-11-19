"""Compare actual AdamW moments, not just final parameter tensors."""

import pytest
import torch

from rebuild_gpt.adamw_state_parity import (
    compare_adamw_moments, named_adamw_moments,
)
from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.optimizers import configure_adamw
from test_grouped_step_parity import GroupedReference


def _models_and_optimizers():
    local = GPTLanguageModel(GPTConfig(7, 4, 8, 2, 1, attention_impl="fused"))
    upstream = GroupedReference(local)
    local_opt = configure_adamw(local, learning_rate=0.001)
    ref_opt = upstream.configure_optimizers(
        weight_decay=0.1, learning_rate=0.001, betas=(0.9, 0.95),
        device_type="cpu",
    )
    return local, upstream, local_opt, ref_opt


def _one_step(local, upstream, local_opt, ref_opt):
    x = torch.tensor([[0, 1, 2, 3]])
    y = torch.tensor([[1, 2, 3, -1]])
    local.forward_nanogpt(x, y)[1].backward()
    upstream(x, y)[1].backward()
    local_opt.step()
    ref_opt.step()


def test_same_adamw_moments_compare_equal_after_update():
    local, upstream, local_opt, ref_opt = _models_and_optimizers()
    _one_step(local, upstream, local_opt, ref_opt)
    result = compare_adamw_moments(local, upstream, local_opt, ref_opt)
    assert result["matched_optimizer_states"] == len(list(local.parameters()))
    assert result["max_exp_avg_error"] < 1e-6
    assert result["max_exp_avg_sq_error"] < 1e-6
    assert result["max_step_error"] == 0.0
    snapshot = named_adamw_moments(local, local_opt)
    first = next(iter(snapshot))
    assert snapshot[first]["step"].item() == 1


def test_modified_second_moment_detects_difference():
    local, upstream, local_opt, ref_opt = _models_and_optimizers()
    _one_step(local, upstream, local_opt, ref_opt)
    param = next(iter(ref_opt.state))
    ref_opt.state[param]["exp_avg_sq"].add_(1.0)
    with pytest.raises(AssertionError, match="AdamW exp_avg_sq mismatch"):
        compare_adamw_moments(local, upstream, local_opt, ref_opt)


def test_state_is_unavailable_before_first_optimizer_step():
    local, upstream, local_opt, ref_opt = _models_and_optimizers()
    with pytest.raises(ValueError, match="Missing AdamW moment"):
        named_adamw_moments(local, local_opt)
