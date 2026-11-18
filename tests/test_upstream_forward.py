import pytest
import torch
from torch.nn import functional as F

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel


@pytest.mark.parametrize("bias", [True, False])
def test_inference_returns_only_last_position_logits(bias):
    model = GPTLanguageModel(GPTConfig(7, 5, 8, 2, 2, attention_impl="fused", bias=bias))
    model.eval()
    x = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    with torch.no_grad():
        all_logits = model(x)
        final_logits, loss = model.forward_nanogpt(x)
    assert final_logits.shape == (2, 1, 7)
    assert torch.equal(final_logits, all_logits[:, -1:, :])
    assert loss is None


@pytest.mark.parametrize("bias", [True, False])
def test_training_contract_matches_masked_reference_cross_entropy(bias):
    model = GPTLanguageModel(GPTConfig(7, 5, 8, 2, 2, attention_impl="fused", bias=bias))
    model.eval()
    x = torch.tensor([[0, 1, 2], [2, 3, 4]])
    y = torch.tensor([[1, 2, -1], [3, -1, 5]])
    logits, loss = model.forward_nanogpt(x, y)
    assert loss is not None
    assert torch.equal(logits, model(x))
    expected = F.cross_entropy(logits.reshape(-1, 7), y.reshape(-1), ignore_index=-1)
    assert torch.allclose(loss, expected)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_existing_forward_and_loss_still_return_full_predictions():
    model = GPTLanguageModel(GPTConfig(7, 4, 8, 2, 1))
    x = torch.tensor([[0, 1, 2]])
    y = torch.tensor([[1, 2, 3]])
    assert model(x).shape == (1, 3, 7)
    assert torch.allclose(model.forward_nanogpt(x, y)[1], model.loss(x, y))


def test_reference_forward_rejects_wrong_target_shape():
    model = GPTLanguageModel(GPTConfig(7, 4, 8, 2, 1))
    with pytest.raises(ValueError, match="targets"):
        model.forward_nanogpt(torch.tensor([[1, 2]]), torch.tensor([[1]]))
