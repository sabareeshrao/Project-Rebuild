"""Per-window GPT NLLs agree with old per-batch loss estimates."""

import copy

import pytest
import torch

from rebuild_gpt.balanced_evaluation import prepare_balanced_evaluation_index_plan
from rebuild_gpt.config import GPTConfig
from rebuild_gpt.eval_index_plan import prepare_evaluation_index_plan
from rebuild_gpt.evaluation_samples import collect_split_losses
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.window_loss_evaluation import collect_window_losses


def model():
    torch.manual_seed(225)
    return GPTLanguageModel(
        GPTConfig(7, 6, 8, 2, 1, dropout=0.25, attention_impl="sdpa")
    )


@pytest.mark.parametrize("policy", ("replacement", "balanced"))
def test_per_window_losses_reconstruct_batch_means_and_preserve_state(policy):
    first = model().train()
    tokens = torch.arange(180) % 7
    args = dict(batch_size=3, block_size=5, eval_batches=2, seed=225)
    plan = (
        prepare_balanced_evaluation_index_plan(
            tokens, region_count=4, **args,
        ) if policy == "balanced" else prepare_evaluation_index_plan(tokens, **args)
    )
    rng = torch.get_rng_state().clone()
    weights_before = {key: value.clone() for key, value in first.state_dict().items()}
    report = collect_window_losses(first, plan)
    reference = collect_split_losses(first, plan)
    assert len(report.starts) == len(report.per_window_losses) == 6
    assert len(report.batch_means) == 2
    assert report.batch_means == pytest.approx(reference, abs=1e-6)
    assert report.sampled_window_mean == pytest.approx(
        sum(reference) / len(reference), abs=1e-6,
    )
    assert first.training
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(weights_before[k], value)
               for k, value in first.state_dict().items())


def test_identical_model_copies_give_identical_window_losses():
    first = model()
    second = copy.deepcopy(first)
    plan = prepare_evaluation_index_plan(
        torch.arange(200) % 7, batch_size=2, block_size=5,
        eval_batches=2, seed=2,
    )
    a = collect_window_losses(first, plan)
    b = collect_window_losses(second, plan)
    assert a == b


def test_large_context_rejected_without_mutating_model():
    first = model().train()
    plan = prepare_evaluation_index_plan(
        torch.arange(100) % 7, batch_size=2, block_size=7,
        eval_batches=2, seed=1,
    )
    with pytest.raises(ValueError, match="block_size"):
        collect_window_losses(first, plan)
    assert first.training


def test_start_order_is_preserved_across_batches():
    first = model().eval()
    plan = prepare_balanced_evaluation_index_plan(
        torch.arange(120) % 7, batch_size=2, block_size=4,
        eval_batches=3, seed=2, region_count=3,
    )
    result = collect_window_losses(first, plan)
    assert result.starts == tuple(int(s) for batch in plan.starts for s in batch)
    assert len(result.per_window_losses) == 6
