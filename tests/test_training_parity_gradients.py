import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.training_parity import capture_upstream_named_gradients


def _model():
    return GPTLanguageModel(GPTConfig(
        vocab_size=7, block_size=4, n_embd=8, n_head=2,
        n_layer=1, attention_impl="fused",
    ))


def test_snapshot_captures_unique_tied_parameter_gradients():
    torch.manual_seed(11)
    model = _model()
    assert model.tokens.lookup.weight is model.head.linear.weight
    inputs = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    targets = torch.tensor([[1, 2, 3, -1]], dtype=torch.long)
    _, loss = model.forward_nanogpt(inputs, targets)
    loss.backward()

    snapshot = capture_upstream_named_gradients(model)
    assert "transformer.wte.weight" in snapshot
    assert "lm_head.weight" not in snapshot  # tied to transformer.wte.weight
    assert len(snapshot) == len(list(model.parameters()))
    assert all(torch.isfinite(g).all() for g in snapshot.values())
    original = model.tokens.lookup.weight.grad.clone()
    model.zero_grad(set_to_none=True)
    assert torch.equal(snapshot["transformer.wte.weight"], original)


def test_cannot_snapshot_before_backward():
    with pytest.raises(ValueError, match="backward"):
        capture_upstream_named_gradients(_model())


def test_unsupported_attention_is_rejected():
    model = GPTLanguageModel(GPTConfig(7, 4, 8, 2, 1))
    with pytest.raises(ValueError, match="fused/SDPA"):
        capture_upstream_named_gradients(model)
