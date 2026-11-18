import pytest
import torch

from rebuild_gpt.checkpoint import load_model_checkpoint, save_model_checkpoint
from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel


def test_checkpoint_roundtrip_preserves_exact_logits_and_tied_weights(tmp_path):
    torch.manual_seed(9)
    model = GPTLanguageModel(GPTConfig(11, 5, 8, 2, 2, attention_impl="fused"))
    model.eval()
    inputs = torch.tensor([[1, 3, 4]])
    with torch.no_grad():
        expected = model(inputs)
    path = tmp_path / "nested" / "weights.pt"
    save_model_checkpoint(model, path)
    loaded = load_model_checkpoint(path)
    assert loaded.config == model.config
    assert loaded.tokens.lookup.weight is loaded.head.linear.weight
    with torch.no_grad():
        assert torch.equal(expected, loaded(inputs))
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), loaded.parameters()))


def test_checkpoint_rejects_wrong_version(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"format_version": 400}, path)
    with pytest.raises(ValueError, match="format_version"):
        load_model_checkpoint(path)


def test_checkpoint_strictly_rejects_missing_weight(tmp_path):
    model = GPTLanguageModel(GPTConfig(7, 3, 8, 2, 1))
    path = tmp_path / "broken.pt"
    save_model_checkpoint(model, path)
    obj = torch.load(path, weights_only=True)
    obj["model"].pop("positions.lookup.weight")
    torch.save(obj, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_model_checkpoint(path)
