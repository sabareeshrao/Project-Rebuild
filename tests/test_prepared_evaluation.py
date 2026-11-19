"""Cached and regenerated validation batches give the same model losses."""

import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.eval_cache import prepare_evaluation_batches
from rebuild_gpt.evaluation import (
    estimate_split_loss, estimate_train_val_loss,
    evaluate_prepared_split_loss, evaluate_prepared_train_val_loss,
)
from rebuild_gpt.gpt import GPTLanguageModel


def make_model():
    torch.manual_seed(162)
    return GPTLanguageModel(
        GPTConfig(7, 5, 8, 2, 1, dropout=0.3, attention_impl="sdpa")
    )


def test_prepared_loss_matches_legacy_and_restores_training_mode_and_rng():
    model = make_model().train()
    tokens = torch.arange(100) % 7
    plan = prepare_evaluation_batches(
        tokens, batch_size=3, block_size=4, eval_batches=3, seed=44,
    )
    prior_rng = torch.get_rng_state().clone()
    original = estimate_split_loss(
        model, tokens, batch_size=3, block_size=4, eval_batches=3, seed=44,
    )
    actual = evaluate_prepared_split_loss(model, plan)
    assert actual == original
    assert model.training
    assert torch.equal(prior_rng, torch.get_rng_state())
    assert all(param.grad is None for param in model.parameters())
    assert evaluate_prepared_split_loss(model, plan) == actual


def test_independent_training_validation_plans_match_reference():
    model = make_model().eval()
    train = torch.arange(120) % 7
    val = (torch.arange(80) * 3) % 7
    a = prepare_evaluation_batches(
        train, batch_size=3, block_size=4, eval_batches=2, seed=8,
    )
    b = prepare_evaluation_batches(
        val, batch_size=3, block_size=4, eval_batches=2, seed=9,
    )
    actual = evaluate_prepared_train_val_loss(model, a, b)
    reference = estimate_train_val_loss(
        model, train, val, batch_size=3,
        block_size=4, eval_batches=2, seed=8,
    )
    assert actual == reference
    assert not model.training


def test_cached_validation_tracks_new_weights_without_recreating_token_ids():
    model = make_model()
    tokens = torch.arange(100) % 7
    plan = prepare_evaluation_batches(
        tokens, batch_size=3, block_size=3, eval_batches=2, seed=12,
    )
    before = evaluate_prepared_split_loss(model, plan)
    with torch.no_grad():
        model.lm_head.weight.add_(0.05 * torch.randn_like(model.lm_head.weight))
    after = evaluate_prepared_split_loss(model, plan)
    assert before != after


def test_context_exceeding_model_is_rejected_without_changing_mode():
    model = make_model().train()
    plan = prepare_evaluation_batches(
        torch.arange(100) % 7, batch_size=2, block_size=6,
        eval_batches=2, seed=1,
    )
    with pytest.raises(ValueError, match="block_size"):
        evaluate_prepared_split_loss(model, plan)
    assert model.training


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_plan_evaluates_cuda_model_without_leaking_cpu_rng():
    model = make_model().to("cuda").train()
    tokens = torch.arange(100) % 7
    plan = prepare_evaluation_batches(
        tokens, batch_size=2, block_size=4, eval_batches=2, seed=1,
    )
    prior = torch.get_rng_state().clone()
    a = evaluate_prepared_split_loss(model, plan)
    b = estimate_split_loss(
        model, tokens, batch_size=2, block_size=4, eval_batches=2, seed=1,
    )
    assert a == b
    assert model.training and torch.equal(prior, torch.get_rng_state())
