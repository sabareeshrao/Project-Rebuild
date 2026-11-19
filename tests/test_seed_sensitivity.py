"""Seed sensitivity is deterministic and never trains the model."""

import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.evaluation import estimate_split_loss
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.seed_sensitivity import evaluate_seed_sensitivity


def make_model():
    torch.manual_seed(180)
    return GPTLanguageModel(
        GPTConfig(9, 5, 8, 2, 1, dropout=0.3, attention_impl="sdpa")
    )


def test_seed_sweep_matches_individual_evaluations_and_restores_rng():
    model = make_model().train()
    tokens = torch.arange(120) % 9
    weights = {key: val.clone() for key, val in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    options = dict(seeds=(10, 12, 15), batch_size=3, block_size=4, eval_batches=2)
    first = evaluate_seed_sensitivity(model, tokens, **options)
    second = evaluate_seed_sensitivity(model, tokens, **options)
    assert first == second
    assert first.per_seed_means == tuple(
        estimate_split_loss(
            model, tokens, seed=seed, batch_size=3,
            block_size=4, eval_batches=2,
        ) for seed in (10, 12, 15)
    )
    assert first.summary.mean == pytest.approx(
        sum(first.per_seed_means) / len(first.per_seed_means)
    )
    assert first.summary.count == 3
    assert model.training and torch.equal(rng, torch.get_rng_state())
    for key, value in model.state_dict().items():
        assert torch.equal(weights[key], value)


def test_single_seed_has_undefined_seed_spread():
    result = evaluate_seed_sensitivity(
        make_model(), torch.arange(120) % 9,
        seeds=(1,), batch_size=2, block_size=4,
    )
    assert result.summary.sample_stdev is None
    assert result.summary.standard_error is None


@pytest.mark.parametrize("bad", [
    [], (2, 2), (-1, 1), (True, 2), (1.2,), "1,2", None,
])
def test_invalid_seed_sets_are_rejected(bad):
    with pytest.raises(ValueError, match="seeds"):
        evaluate_seed_sensitivity(
            make_model(), torch.arange(120) % 9,
            seeds=bad, batch_size=2, block_size=4,
        )


def test_invalid_batch_counts_rejected():
    with pytest.raises(ValueError, match="eval_batches"):
        evaluate_seed_sensitivity(
            make_model(), torch.arange(120) % 9,
            seeds=(1, 2), batch_size=2, block_size=4, eval_batches=0,
        )
