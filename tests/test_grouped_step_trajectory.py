"""Repeated optimizer updates must reuse AdamW moments across steps."""

import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.grouped_step_parity import compare_grouped_adamw_trajectory
from rebuild_gpt.gpt import GPTLanguageModel
from test_grouped_step_parity import GroupedReference


def _models():
    model = GPTLanguageModel(GPTConfig(
        7, 4, 8, 2, 1, attention_impl="fused", dropout=0.1,
    ))
    return model, GroupedReference(model)


def test_three_persistent_steps_match_and_keep_evaluation_deterministic():
    ours, reference = _models()
    ours.train()
    reference.train()
    x = torch.tensor([[0, 1, 2, 3]])
    y = torch.tensor([[1, 2, 3, -1]])
    report = compare_grouped_adamw_trajectory(
        ours, reference, x, y, steps=3,
    )
    assert report["steps"] == 3
    assert [s["step"] for s in report["history"]] == [1, 2, 3]
    assert all(s["max_updated_weight_error"] < 1e-6 for s in report["history"])
    assert all(s["max_gradient_error"] < 1e-6 for s in report["history"])
    assert ours.training and reference.training


def test_negative_and_noninteger_step_counts_rejected():
    ours, reference = _models()
    x = torch.tensor([[0, 1]])
    y = torch.tensor([[1, -1]])
    for steps in (0, -2, 1.5, True):
        with pytest.raises(ValueError, match="steps"):
            compare_grouped_adamw_trajectory(
                ours, reference, x, y, steps=steps,
            )


def test_input_validation_before_synchronizing_weights():
    ours, reference = _models()
    with pytest.raises(ValueError, match="\[B,T\]"):
        compare_grouped_adamw_trajectory(
            ours, reference, torch.tensor([0, 1]), torch.tensor([1, -1]),
        )
