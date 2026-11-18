import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.text_checkpoint import load_text_checkpoint, save_text_checkpoint
from rebuild_gpt.text_generation import generate_text
from rebuild_gpt.tokenizer import CharTokenizer


def make_pair():
    tokenizer = CharTokenizer("hello world.\n")
    torch.manual_seed(2)
    model = GPTLanguageModel(GPTConfig(
        tokenizer.vocab_size, 8, 8, 2, 2, attention_impl="fused",
    )).eval()
    return model, tokenizer


def test_text_checkpoint_roundtrip_restores_logits_and_vocabulary(tmp_path):
    model, tok = make_pair()
    prompt = "hello"
    ids = torch.tensor([tok.encode(prompt)])
    with torch.no_grad():
        before = model(ids)
    path = tmp_path / "nested" / "text.pt"
    save_text_checkpoint(model, tok, path)
    loaded, loaded_tokenizer = load_text_checkpoint(path)
    assert loaded_tokenizer.stoi == tok.stoi
    assert loaded.config == model.config
    assert loaded.tokens.lookup.weight is loaded.head.linear.weight
    with torch.no_grad():
        assert torch.equal(loaded(ids), before)
    a = generate_text(model, tok, prompt, 4, generator=torch.Generator().manual_seed(7))
    b = generate_text(loaded, loaded_tokenizer, prompt, 4, generator=torch.Generator().manual_seed(7))
    assert a == b


def test_cannot_save_mismatched_tokenizer(tmp_path):
    model, _ = make_pair()
    with pytest.raises(ValueError, match="vocab_size"):
        save_text_checkpoint(model, CharTokenizer("ab"), tmp_path / "bad.pt")


def test_loader_rejects_checkpoint_with_wrong_vocabulary_size(tmp_path):
    model, tok = make_pair()
    path = tmp_path / "bad.pt"
    save_text_checkpoint(model, tok, path)
    payload = torch.load(path, weights_only=True)
    payload["vocab"] = ["a", "b"]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="vocab_size"):
        load_text_checkpoint(path)


def test_loader_rejects_missing_model_weight(tmp_path):
    model, tok = make_pair()
    path = tmp_path / "bad.pt"
    save_text_checkpoint(model, tok, path)
    payload = torch.load(path, weights_only=True)
    payload["model"].pop("positions.lookup.weight")
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_text_checkpoint(path)


def test_loader_rejects_unsupported_version(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"format_version": 123}, path)
    with pytest.raises(ValueError, match="format_version"):
        load_text_checkpoint(path)
