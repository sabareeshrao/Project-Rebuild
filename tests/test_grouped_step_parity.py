"""Check grouped AdamW against a local reference adapter, not upstream code."""

import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.grouped_step_parity import compare_one_grouped_adamw_step
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.optimizers import configure_adamw
from test_training_parity_comparison import RenamedLocalReference


class GroupedReference(RenamedLocalReference):
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        return configure_adamw(
            self, learning_rate=learning_rate, weight_decay=weight_decay,
            betas=betas, device_type=device_type,
        )


def _models():
    ours = GPTLanguageModel(GPTConfig(7, 4, 8, 2, 1, attention_impl="fused"))
    return ours, GroupedReference(ours)


def test_one_grouped_step_matches_parameters_and_gradients():
    ours, reference = _models()
    x = torch.tensor([[0, 1, 2, 3]])
    y = torch.tensor([[1, 2, 3, -1]])
    report = compare_one_grouped_adamw_step(ours, reference, x, y)
    assert report["matched_parameters"] > 0
    assert report["decay_parameters"] > 0
    assert report["no_decay_parameters"] > 0
    assert report["max_gradient_error"] < 1e-6
    assert report["max_updated_weight_error"] < 1e-6
    assert report["largest_local_parameter_change"] > 0


def test_reference_must_expose_original_optimizer_api():
    ours, reference = _models()
    reference.configure_optimizers = None
    with pytest.raises(ValueError, match="configure_optimizers"):
        compare_one_grouped_adamw_step(
            ours, reference, torch.tensor([[0, 1]]), torch.tensor([[1, -1]]),
        )


@pytest.mark.parametrize("bad", [
    {"learning_rate": 0}, {"learning_rate": float("inf")},
    {"weight_decay": -1}, {"weight_decay": float("nan")},
])
def test_bad_optimizer_settings_are_rejected(bad):
    ours, reference = _models()
    with pytest.raises(ValueError):
        compare_one_grouped_adamw_step(
            ours, reference, torch.tensor([[0, 1]]),
            torch.tensor([[1, -1]]), **bad,
        )
