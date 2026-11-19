"""All evaluation backends expose exactly the same per-batch losses."""

import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.eval_cache import prepare_evaluation_batches
from rebuild_gpt.eval_index_plan import prepare_evaluation_index_plan
from rebuild_gpt.evaluation import estimate_split_loss
from rebuild_gpt.evaluation_samples import collect_split_losses
from rebuild_gpt.evaluation_stats import summarize_losses
from rebuild_gpt.gpt import GPTLanguageModel


def make_model():
    torch.manual_seed(179)
    return GPTLanguageModel(
        GPTConfig(13, 6, 8, 2, 1, dropout=0.3, attention_impl="sdpa")
    )


@pytest.mark.parametrize("batch_size,context,count", [(1, 2, 1), (3, 4, 3)])
def test_three_backends_share_exact_loss_samples(batch_size, context, count):
    model = make_model().train()
    tokens = torch.arange(140) % 13
    kwargs = dict(batch_size=batch_size, block_size=context,
                  eval_batches=count, seed=29)
    before = torch.get_rng_state().clone()
    regenerated = collect_split_losses(model, tokens, **kwargs)
    cached = collect_split_losses(
        model, prepare_evaluation_batches(tokens, **kwargs)
    )
    indices = collect_split_losses(
        model, prepare_evaluation_index_plan(tokens, **kwargs)
    )
    assert regenerated == cached == indices
    assert sum(regenerated) / len(regenerated) == estimate_split_loss(
        model, tokens, **kwargs
    )
    assert summarize_losses(indices).count == count
    assert model.training and torch.equal(before, torch.get_rng_state())
    assert all(p.grad is None for p in model.parameters())


def test_eval_mode_remains_eval_and_samples_are_repeatable():
    model = make_model().eval()
    tokens = torch.arange(140) % 13
    first = collect_split_losses(
        model, tokens, batch_size=2, block_size=4, eval_batches=3, seed=1,
    )
    second = collect_split_losses(
        model, tokens, batch_size=2, block_size=4, eval_batches=3, seed=1,
    )
    assert first == second and not model.training


def test_invalid_context_rejected_without_changing_train_mode():
    model = make_model().train()
    with pytest.raises(ValueError, match="block_size"):
        collect_split_losses(
            model, torch.arange(140), batch_size=2, block_size=7,
        )
    assert model.training


def test_invalid_raw_batch_count_rejected():
    with pytest.raises(ValueError, match="eval_batches"):
        collect_split_losses(
            make_model(), torch.arange(60),
            batch_size=2, block_size=3, eval_batches=0,
        )
