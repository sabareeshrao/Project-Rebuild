import pytest
import torch

from rebuild_gpt.config import GPTConfig
from rebuild_gpt.gpt import GPTLanguageModel
from rebuild_gpt.text_generation import generate_text
from rebuild_gpt.tokenizer import CharTokenizer


def make_pair(dropout=0.0):
    tok = CharTokenizer("abab.\n")
    torch.manual_seed(7)
    model = GPTLanguageModel(GPTConfig(
        tok.vocab_size, block_size=6, n_embd=8, n_head=2,
        n_layer=1, dropout=dropout, attention_impl="fused",
    ))
    return model, tok


def test_text_generation_preserves_prompt_and_decodes_generated_ids():
    model, tok = make_pair()
    prompt = "ab"
    seed = 4
    out = generate_text(model, tok, prompt, 5, generator=torch.Generator().manual_seed(seed))
    ids = model.generate(torch.tensor([tok.encode(prompt)]), 5, generator=torch.Generator().manual_seed(seed))
    assert out == tok.decode(ids[0].tolist())
    assert out.startswith(prompt)
    assert len(out) == len(prompt) + 5


def test_generate_text_is_seeded_and_restores_training_mode():
    model, tok = make_pair(dropout=0.3)
    model.train()
    a = generate_text(model, tok, "a", 4, generator=torch.Generator().manual_seed(9))
    assert model.training
    b = generate_text(model, tok, "a", 4, generator=torch.Generator().manual_seed(9))
    assert a == b


def test_zero_new_tokens_returns_exact_prompt():
    model, tok = make_pair()
    assert generate_text(model, tok, "ab", 0) == "ab"


def test_invalid_or_unseen_prompt_rejected():
    model, tok = make_pair()
    with pytest.raises(ValueError, match="prompt"):
        generate_text(model, tok, "", 2)
    with pytest.raises(ValueError, match="Character"):
        generate_text(model, tok, "z", 2)


def test_tokenizer_model_vocab_mismatch_rejected():
    model, _ = make_pair()
    with pytest.raises(ValueError, match="vocab_size"):
        generate_text(model, CharTokenizer("abc"), "a", 2)


def test_training_mode_restored_when_sampling_fails():
    model, tok = make_pair(dropout=0.3)
    model.train()
    with pytest.raises(ValueError, match="temperature"):
        generate_text(model, tok, "a", 2, temperature=0)
    assert model.training
