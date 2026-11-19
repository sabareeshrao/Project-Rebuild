"""Compact index plans should not change losses, dropout or model weights."""

import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.eval_cache import prepare_train_val_evaluation_batches
from rebuild_gpt.eval_index_plan import (
    prepare_evaluation_index_plan, prepare_train_val_index_plans,
)
from rebuild_gpt.evaluation import (
    estimate_split_loss, estimate_train_val_loss,
    evaluate_prepared_train_val_loss,
    evaluate_index_plan_split_loss, evaluate_index_plan_train_val_loss,
)
from rebuild_gpt.gpt import GPTLanguageModel


def make_model():
    torch.manual_seed(175)
    return GPTLanguageModel(
        GPTConfig(11, 6, 8, 2, 1, dropout=0.2, attention_impl="sdpa")
    )


def test_index_evaluation_matches_regenerated_loss_exactly_and_restores_state():
    model = make_model().train()
    tokens = torch.arange(120, dtype=torch.long) % 11
    plan = prepare_evaluation_index_plan(
        tokens, batch_size=3, block_size=5, eval_batches=3, seed=175,
    )
    before_rng = torch.get_rng_state().clone()
    original = estimate_split_loss(
        model, tokens, batch_size=3, block_size=5, eval_batches=3, seed=175,
    )
    actual = evaluate_index_plan_split_loss(model, plan)
    assert actual == original
    assert model.training
    assert torch.equal(before_rng, torch.get_rng_state())
    assert all(p.grad is None for p in model.parameters())
    assert evaluate_index_plan_split_loss(model, plan) == original


def test_three_evaluation_backends_agree_on_both_splits():
    model = make_model().eval()
    train = torch.arange(110) % 11
    val = (torch.arange(76) * 3) % 11
    plans = prepare_train_val_index_plans(
        train, val, batch_size=2, block_size=4, eval_batches=3, seed=200,
    )
    materialized = prepare_train_val_evaluation_batches(
        train, val, batch_size=2, block_size=4, eval_batches=3, seed=200,
    )
    reference = estimate_train_val_loss(
        model, train, val, batch_size=2, block_size=4,
        eval_batches=3, seed=200,
    )
    assert evaluate_index_plan_train_val_loss(model, *plans) == reference
    assert evaluate_prepared_train_val_loss(model, *materialized) == reference
    assert model.training is False


def test_index_evaluation_updates_losses_with_current_weights():
    model = make_model()
    tokens = torch.arange(120) % 11
    plan = prepare_evaluation_index_plan(
        tokens, batch_size=2, block_size=3, eval_batches=2,
    )
    before = evaluate_index_plan_split_loss(model, plan)
    with torch.no_grad():
        model.head.linear.weight.add_(torch.randn_like(model.head.linear.weight) * 0.2)
    assert evaluate_index_plan_split_loss(model, plan) != before


def test_invalid_context_preserves_mode():
    model = make_model().train()
    plan = prepare_evaluation_index_plan(
        torch.arange(100), batch_size=2, block_size=7,
    )
    with pytest.raises(ValueError, match="block_size"):
        evaluate_index_plan_split_loss(model, plan)
    assert model.training


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cpu_index_plan_runs_cuda_gpt_with_unchanged_loss():
    model = make_model().to("cuda").train()
    cpu = torch.arange(120) % 11
    plan = prepare_evaluation_index_plan(
        cpu, batch_size=2, block_size=4, eval_batches=2, seed=7,
    )
    actual = evaluate_index_plan_split_loss(model, plan)
    reference = estimate_split_loss(
        model, cpu, batch_size=2, block_size=4, eval_batches=2, seed=7,
    )
    assert actual == reference and model.training
