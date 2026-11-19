"""Paired model comparison uses identical validation windows without training."""

import copy

import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.evaluation import estimate_split_loss
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.paired_evaluation import compare_paired_validation


def make_model(vocab=9):
    torch.manual_seed(182)
    return GPTLanguageModel(
        GPTConfig(vocab, 5, 8, 2, 1, dropout=0.2, attention_impl="sdpa")
    )


def test_identical_models_have_exactly_zero_paired_differences_and_no_rng_change():
    baseline = make_model().train()
    candidate = copy.deepcopy(baseline).eval()
    rng = torch.get_rng_state().clone()
    tokens = torch.arange(120) % 9
    result = compare_paired_validation(
        baseline, candidate, tokens, seeds=(11, 12, 13),
        batch_size=3, block_size=4, eval_batches=2,
    )
    assert result.candidate_minus_baseline == (0.0, 0.0, 0.0)
    assert result.delta_summary.mean == 0
    assert result.delta_summary.sample_stdev == 0
    assert baseline.training and not candidate.training
    assert torch.equal(rng, torch.get_rng_state())
    assert all(p.grad is None for p in candidate.parameters())


def test_paired_result_matches_separate_reference_evaluations():
    baseline = make_model()
    candidate = copy.deepcopy(baseline)
    with torch.no_grad():
        candidate.head.linear.weight.add_(
            0.1 * torch.randn_like(candidate.head.linear.weight)
        )
    tokens = torch.arange(120) % 9
    seeds = (2, 3, 6)
    result = compare_paired_validation(
        baseline, candidate, tokens,
        seeds=seeds, batch_size=2, block_size=4, eval_batches=3,
    )
    assert any(abs(delta) > 0 for delta in result.candidate_minus_baseline)
    for seed, a, b, delta in zip(
        result.seeds, result.baseline_means, result.candidate_means,
        result.candidate_minus_baseline,
    ):
        options = dict(batch_size=2, block_size=4, eval_batches=3, seed=seed)
        assert a == estimate_split_loss(baseline, tokens, **options)
        assert b == estimate_split_loss(candidate, tokens, **options)
        assert delta == pytest.approx(b - a)


def test_exactly_one_index_plan_is_built_per_seed(monkeypatch):
    import rebuild_gpt.paired_evaluation as paired

    original = paired.prepare_evaluation_index_plan
    calls = []

    def spy(tokens, **kwargs):
        calls.append(kwargs["seed"])
        return original(tokens, **kwargs)

    monkeypatch.setattr(paired, "prepare_evaluation_index_plan", spy)
    model = make_model()
    compare_paired_validation(
        model, copy.deepcopy(model), torch.arange(120) % 9,
        seeds=(7, 8), batch_size=2, block_size=3,
    )
    assert calls == [7, 8]


@pytest.mark.parametrize("seeds", [[], (1, 1), (-1,), (True,), "1,2"])
def test_invalid_seed_sets_rejected(seeds):
    model = make_model()
    with pytest.raises(ValueError, match="seeds"):
        compare_paired_validation(
            model, model, torch.arange(100) % 9,
            seeds=seeds, batch_size=2, block_size=3,
        )


def test_vocab_mismatch_rejected():
    with pytest.raises(ValueError, match="vocabulary"):
        compare_paired_validation(
            make_model(9), make_model(10), torch.arange(100) % 9,
            seeds=(1, 2), batch_size=2, block_size=3,
        )
