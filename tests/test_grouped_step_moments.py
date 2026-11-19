"""Persistent grouped trajectories check optimizer state after each step."""

from unittest.mock import patch

import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.grouped_step_parity import compare_grouped_adamw_trajectory
from test_grouped_step_parity import GroupedReference


def _models():
    local = GPTLanguageModel(GPTConfig(
        7, 4, 8, 2, 1, attention_impl="fused",
    ))
    return local, GroupedReference(local)


def test_trajectory_records_all_three_adamw_state_errors():
    local, upstream = _models()
    x = torch.tensor([[0, 1, 2, 3]])
    y = torch.tensor([[1, 2, 3, -1]])
    result = compare_grouped_adamw_trajectory(local, upstream, x, y, steps=3)
    assert len(result["history"]) == 3
    for step, row in enumerate(result["history"], start=1):
        assert row["step"] == step
        assert row["matched_optimizer_states"] == len(list(local.parameters()))
        assert row["max_step_error"] == 0.0
        assert row["max_exp_avg_error"] < 1e-6
        assert row["max_exp_avg_sq_error"] < 1e-6


def test_trajectory_does_not_succeed_if_moment_state_mismatch():
    local, upstream = _models()
    x = torch.tensor([[0, 1, 2, 3]])
    y = torch.tensor([[1, 2, 3, -1]])
    with patch(
        "rebuild_gpt.adamw_state_parity.compare_adamw_moments",
        side_effect=AssertionError("moment drift"),
    ):
        with pytest.raises(AssertionError, match="moment drift"):
            compare_grouped_adamw_trajectory(local, upstream, x, y, steps=2)
