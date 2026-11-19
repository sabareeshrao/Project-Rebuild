"""Top-k, temperature and context cropping determine sampling probabilities."""

import math

import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.next_token_distribution import inspect_next_token_distribution


class FixedLogitGPT(GPTLanguageModel):
    def forward_nanogpt(self, token_ids, targets=None):
        scores = torch.tensor([0.0, 3.0, 1.0, -2.0])
        return scores.reshape(1, 1, 4), None


def _model():
    return FixedLogitGPT(GPTConfig(
        4, 3, 8, 2, 1, dropout=0.3, attention_impl="fused",
    ))


def test_top_two_distribution_has_exact_support_and_normalization():
    model = _model()
    model.train()
    previous = torch.get_rng_state().clone()
    report = inspect_next_token_distribution(
        model, torch.tensor([[0, 1, 2, 3, 0]]), top_k=2,
    )
    assert report["prompt_length"] == 5
    assert report["context_length"] == 3
    assert report["allowed_ids"] == [1, 2]
    assert report["peak_id"] == 1
    assert report["candidate_count"] == 2
    assert math.isclose(sum(report["probabilities"]), 1.0, abs_tol=1e-6)
    assert report["probabilities"][0] == 0.0
    assert report["probabilities"][3] == 0.0
    assert model.training
    assert torch.equal(torch.get_rng_state(), previous)


def test_top_one_is_a_one_token_distribution_with_zero_entropy():
    result = inspect_next_token_distribution(
        _model(), torch.tensor([[0, 1]]), top_k=1,
    )
    assert result["allowed_ids"] == [1]
    assert result["probabilities"][1] == 1.0
    assert result["entropy_nats"] == 0.0


def test_temperature_changes_entropy_without_changing_top_k_support():
    prompt = torch.tensor([[0, 1]])
    cold = inspect_next_token_distribution(
        _model(), prompt, temperature=0.2, top_k=2,
    )
    warm = inspect_next_token_distribution(
        _model(), prompt, temperature=2.0, top_k=2,
    )
    assert cold["allowed_ids"] == warm["allowed_ids"] == [1, 2]
    assert cold["entropy_nats"] < warm["entropy_nats"]


@pytest.mark.parametrize("bad", [
    {"temperature": 0},
    {"temperature": float("inf")},
    {"top_k": 0},
    {"top_k": True},
])
def test_invalid_sampling_controls_are_rejected(bad):
    with pytest.raises(ValueError):
        inspect_next_token_distribution(
            _model(), torch.tensor([[0, 1]]), **bad,
        )


def test_empty_prompt_is_rejected_before_model_forward():
    with pytest.raises(ValueError, match="nonempty"):
        inspect_next_token_distribution(
            _model(), torch.empty((1, 0), dtype=torch.long),
        )
