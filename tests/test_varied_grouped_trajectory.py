"""Parity with different seeded training inputs must preserve AdamW state."""

from unittest.mock import patch

import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.grouped_step_parity import compare_grouped_adamw_trajectory
from test_grouped_step_parity import GroupedReference


def _models():
    ours = GPTLanguageModel(GPTConfig(
        9, 4, 8, 2, 1, attention_impl="fused", dropout=0.1,
    ))
    return ours, GroupedReference(ours)


def _pairs():
    return (
        (torch.tensor([[0, 1, 2, 3]]), torch.tensor([[1, 2, 3, -1]])),
        (torch.tensor([[5, 4, 3, 2]]), torch.tensor([[4, 3, 2, -1]])),
        (torch.tensor([[0, 8, 4, 6]]), torch.tensor([[8, 4, 6, -1]])),
    )


def test_trajectory_uses_a_different_batch_at_each_step():
    ours, upstream = _models()
    pairs = _pairs()
    with patch.object(ours, "forward_nanogpt", wraps=ours.forward_nanogpt) as run:
        report = compare_grouped_adamw_trajectory(
            ours, upstream, *pairs[0], steps=3, batches=pairs,
        )
    assert len(report["history"]) == 3
    assert run.call_count == 3
    assert all(torch.equal(call.args[0], pairs[i][0])
               for i, call in enumerate(run.call_args_list))
    assert all(torch.equal(call.args[1], pairs[i][1])
               for i, call in enumerate(run.call_args_list))
    assert all(s["max_updated_weight_error"] < 1e-6 for s in report["history"])
    assert all(s["max_gradient_error"] < 1e-6 for s in report["history"])


@pytest.mark.parametrize("invalid", [
    [], [(_pairs()[0][0], _pairs()[0][1])],
    [(torch.tensor([0, 1]), torch.tensor([1, 2]))] * 3,
])
def test_bad_step_batch_sequence_fails_before_optimizer_steps(invalid):
    ours, upstream = _models()
    with pytest.raises(ValueError, match="batches|batch pair"):
        compare_grouped_adamw_trajectory(
            ours, upstream, *_pairs()[0], steps=3, batches=invalid,
        )
