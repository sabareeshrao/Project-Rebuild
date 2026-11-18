from pathlib import Path

import torch

from rebuild_gpt.checkpoint import load_model_checkpoint, save_model_checkpoint
from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.nanogpt_keys import export_nanogpt_state_dict, load_nanogpt_state_dict
from rebuild_gpt.training import train_step


def test_trained_fused_gpt_survives_native_and_translated_roundtrips(tmp_path: Path):
    torch.manual_seed(23)
    config = GPTConfig(11, 5, 8, 2, 2, attention_impl="fused")
    trained = GPTLanguageModel(config)
    optimizer = torch.optim.AdamW(trained.parameters(), lr=0.01)
    x = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    y = torch.tensor([[1, 2, 3, 4], [2, 1, 0, 5]])
    train_step(trained, optimizer, x, y)
    trained.eval()
    with torch.no_grad():
        expected = trained(x)

    native_path = tmp_path / "our_checkpoint.pt"
    save_model_checkpoint(trained, native_path)
    native = load_model_checkpoint(native_path)
    translated_path = tmp_path / "nanogpt_style_tensors.pt"
    torch.save(export_nanogpt_state_dict(native), translated_path)
    translated_tensors = torch.load(translated_path, map_location="cpu", weights_only=True)
    imported = GPTLanguageModel(config).eval()
    load_nanogpt_state_dict(imported, translated_tensors)

    with torch.no_grad():
        assert torch.equal(expected, native(x))
        assert torch.equal(expected, imported(x))
    assert imported.tokens.lookup.weight is imported.head.linear.weight
    assert len(translated_tensors) == len(native.state_dict())


def test_educational_mode_still_works_for_native_checkpoint(tmp_path):
    model = GPTLanguageModel(GPTConfig(7, 3, 8, 2, 1, attention_impl="educational"))
    model.eval()
    sample = torch.tensor([[0, 1, 2]])
    path = tmp_path / "educational.pt"
    save_model_checkpoint(model, path)
    loaded = load_model_checkpoint(path)
    with torch.no_grad():
        assert torch.equal(model(sample), loaded(sample))
