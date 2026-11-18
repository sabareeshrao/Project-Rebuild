import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.evaluation import estimate_split_loss, estimate_train_val_loss
from rebuild_gpt.gpt import GPTLanguageModel


def make_model():
    torch.manual_seed(3)
    return GPTLanguageModel(GPTConfig(3, 5, 8, 2, 1, dropout=0.25, attention_impl="fused"))


def test_estimates_are_repeatable_and_preserve_model_mode_and_rng():
    model = make_model().train()
    tokens = torch.tensor([0, 1, 2] * 30)
    before = torch.get_rng_state().clone()
    first = estimate_split_loss(model, tokens, batch_size=3, block_size=4, eval_batches=3, seed=11)
    second = estimate_split_loss(model, tokens, batch_size=3, block_size=4, eval_batches=3, seed=11)
    assert first == second
    assert first > 0
    assert model.training
    assert torch.equal(before, torch.get_rng_state())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_independent_train_val_estimates_have_expected_keys():
    model = make_model().eval()
    train = torch.tensor([0, 1, 2] * 30)
    val = torch.tensor([0, 2, 1] * 20)
    result = estimate_train_val_loss(model, train, val, batch_size=4, block_size=3, eval_batches=2, seed=7)
    assert set(result) == {"train_loss", "validation_loss"}
    assert all(value > 0 for value in result.values())
    assert not model.training


@pytest.mark.parametrize("option,value", [
    ("eval_batches", 0), ("seed", -1), ("batch_size", 0), ("block_size", 0),
])
def test_evaluation_rejects_invalid_arguments(option, value):
    kwargs = dict(batch_size=2, block_size=3, eval_batches=2, seed=1)
    kwargs[option] = value
    with pytest.raises(ValueError, match=option):
        estimate_split_loss(make_model(), torch.tensor([0, 1, 2] * 10), **kwargs)


def test_evaluation_restores_training_mode_on_bad_stream():
    model = make_model().train()
    with pytest.raises(ValueError, match="block_size"):
        estimate_split_loss(model, torch.tensor([0, 1]), batch_size=2, block_size=3)
    assert model.training


def test_cpu_token_stream_can_be_evaluated_against_model_device():
    net = make_model().to("cpu")
    tokens = torch.tensor([0, 1, 2] * 30, dtype=torch.long)
    loss = estimate_split_loss(
        net, tokens, batch_size=3, block_size=4, eval_batches=2, seed=12
    )
    assert isinstance(loss, float) and loss > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_token_stream_can_evaluate_cuda_model_without_rng_leak():
    net = make_model().to("cuda").train()
    tokens = torch.tensor([0, 1, 2] * 30)
    before = torch.get_rng_state().clone()
    loss_a = estimate_split_loss(
        net, tokens, batch_size=3, block_size=4, eval_batches=2, seed=12
    )
    loss_b = estimate_split_loss(
        net, tokens, batch_size=3, block_size=4, eval_batches=2, seed=12
    )
    assert loss_a == loss_b
    assert net.training
    assert torch.equal(before, torch.get_rng_state())
