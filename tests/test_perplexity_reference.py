"""Uniform guessing has perplexity equal to the vocabulary size."""

import math

import pytest
import torch

from rebuild_gpt.perplexity_reference import compare_uniform_perplexity


def test_uniform_logits_have_exact_vocab_sized_perplexity():
    logits = torch.zeros(2, 3, 5)
    targets = torch.tensor([[0, 1, -1], [2, 3, 4]])
    report = compare_uniform_perplexity(logits, targets)
    assert report["supervised_positions"] == 5
    assert report["ignored_positions"] == 1
    assert report["cross_entropy_nats"] == pytest.approx(math.log(5))
    assert report["bits_per_token"] == pytest.approx(math.log2(5))
    assert report["perplexity"] == pytest.approx(5)
    assert report["excess_loss_vs_uniform_nats"] == pytest.approx(0, abs=1e-6)


def test_correct_high_confidence_prediction_beats_uniform():
    logits = torch.tensor([[[0., 8., 0., -2.]]])
    report = compare_uniform_perplexity(logits, torch.tensor([[1]]))
    assert 1 <= report["perplexity"] < 4
    assert report["excess_loss_vs_uniform_nats"] < 0


def test_confidently_wrong_prediction_is_worse_than_uniform():
    logits = torch.tensor([[[0., 8., 0., -2.]]])
    report = compare_uniform_perplexity(logits, torch.tensor([[0]]))
    assert report["perplexity"] > 4
    assert report["excess_loss_vs_uniform_nats"] > 0


@pytest.mark.parametrize("targets", [
    torch.tensor([[-1, -1]]),
    torch.tensor([[0., -1.]]),
    torch.tensor([[0]]),
    torch.tensor([[0, 99]]),
])
def test_invalid_supervised_targets_are_rejected(targets):
    with pytest.raises(ValueError):
        compare_uniform_perplexity(torch.zeros(1, 2, 4), targets)


def test_nonfinite_logits_cannot_produce_a_perplexity_claim():
    with pytest.raises(ValueError, match="finite"):
        compare_uniform_perplexity(
            torch.tensor([[[float("nan"), 0.]]]), torch.tensor([[1]]),
        )
